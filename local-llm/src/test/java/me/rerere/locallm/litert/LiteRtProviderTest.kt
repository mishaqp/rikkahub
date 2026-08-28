package me.rerere.locallm.litert

import com.google.ai.edge.litertlm.ResponseFormat
import kotlinx.serialization.json.Json
import me.rerere.ai.core.ReasoningLevel
import me.rerere.ai.provider.CustomBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-decision-function tests for [LiteRtProvider]'s image-forwarding gate.
 *
 * [decideImageForwarding] guards against a hard native crash. When a vision-capable model is
 * loaded but the device's GPU vision executor fails to initialise, [LiteRtRuntime.ensureLoaded]
 * falls back to a TEXT-ONLY engine (see [LiteRtRuntime.LoadOutcome.visionEnabled] /
 * [LiteRtRuntime.LoadOutcome.visionFellBackToTextOnly]). Forwarding image bytes to an engine
 * that has no vision executor null-derefs inside `liblitertlm_jni.so` -> SIGSEGV. The gate
 * MUST therefore key off the ACTUAL post-load vision state, never the pre-load estimate (which
 * is still `true` at the moment of the doomed first load).
 */
class LiteRtProviderTest {

    @Test
    fun `forwards images when vision is live post-load`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = true,
        )
        assertTrue("vision live -> forward", d.forwardImages)
        assertFalse("nothing dropped when we forward", d.noteImagesDropped)
    }

    @Test
    fun `does NOT forward images when vision fell back to text-only (the crash case)`() {
        // Vision-capable model, but the GPU vision executor failed and the engine loaded
        // text-only. Forwarding here is the SIGSEGV. Drop the images, and flag a user note.
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse("vision not live -> never forward", d.forwardImages)
        assertTrue("user attached images that were dropped -> note them", d.noteImagesDropped)
    }

    @Test
    fun `no note when vision fell back but the user attached no images`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = false,
        )
        assertFalse(d.forwardImages)
        assertFalse("nothing to note when no images were attached", d.noteImagesDropped)
    }

    @Test
    fun `text-only model with a stray image attachment drops silently (no note)`() {
        // The user picked a model that never supported vision. Preserve the prior silent-drop
        // behaviour: no "vision unavailable on this device" note, because the device's vision
        // capability was never the issue here.
        val d = decideImageForwarding(
            modelImageCapable = false,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse(d.forwardImages)
        assertFalse("text-only model is not a device-vision failure", d.noteImagesDropped)
    }

    @Test
    fun `no images with vision live is a no-op`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = false,
        )
        assertTrue(d.forwardImages)
        assertFalse(d.noteImagesDropped)
    }
}

/**
 * Budget tests for the native tool declarations.
 *
 * Tool declarations are prompt text: the chat template renders each one ahead of the
 * conversation. They were the one part of the prefill with no cap, so an assistant with a
 * large enabled tool set pushed tens of thousands of characters of JSON schema into a model
 * whose whole context is a few thousand tokens. The engine does not bounds-check that: it
 * faults inside the native executor rather than returning a context-overflow error.
 */
class LiteRtToolDeclarationBudgetTest {

    private fun budget(contextTokens: Int) =
        LiteRtProvider.toolDeclarationCharBudget(contextTokens)

    @Test
    fun `budget leaves room for system prompt and history`() {
        // 4096t * 4 chars / 2 = 8192 input chars, minus the 500 + 3000 already committed.
        assertEquals(4692, budget(4096))
    }

    @Test
    fun `budget scales with the model's context`() {
        assertTrue(
            "a 32k model must afford far more declarations than a 4k one",
            budget(32768) > budget(4096) * 10,
        )
    }

    @Test
    fun `budget goes non-positive for a context too small to hold any tool`() {
        // 1280t (qwen3 ekv1280) -> 2560 - 3500 < 0, so no tool is declared at all rather
        // than one being forced in over budget.
        assertTrue("tiny context yields no tool room", budget(1280) <= 0)
    }

    @Test
    fun `declarations are taken until the budget is spent and the rest reported`() {
        val costs = listOf(100, 100, 100, 100)
        val charBudget = 250
        var used = 0
        val dropped = mutableListOf<Int>()
        val kept = costs.filterIndexed { index, cost ->
            if (used + cost > charBudget) {
                dropped += index
                false
            } else {
                used += cost
                true
            }
        }
        assertEquals("only whole declarations that fit are kept", listOf(100, 100), kept)
        assertEquals(200, used)
        assertEquals("the overflow is reported, not silently swallowed", listOf(2, 3), dropped)
    }
}

/**
 * The prompt has to be budgeted against the context the ENGINE was configured with, never
 * against the ceiling the model file could theoretically hold. Those are different numbers
 * whenever the catalog allocates less than the file supports, and budgeting against the
 * ceiling let a 32768-token budget build a prompt for an engine holding 4096: on
 * gemma-4-E2B the whole enabled tool set was declared, 31067 chars of it, and the engine
 * rejected the prefill at 7868 tokens.
 */
class LiteRtEngineContextTokensTest {

    @Test
    fun `the allocation wins when the file ceiling is higher`() {
        // gemma-4-E2B: catalog allocates 4096, the file could hold 32768.
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, 32768))
    }

    @Test
    fun `the ceiling wins when the request is higher`() {
        // qwen3_0_6b_mixed_int4 declares 2048; asking for more faults the native executor.
        assertEquals(2048, LiteRtProvider.engineContextTokens(32000, 2048))
    }

    @Test
    fun `an unknown ceiling leaves the request untouched`() {
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, null))
    }

    @Test
    fun `the tool budget derived from the engine size never exceeds the engine`() {
        // The regression this guards: budget(ceiling) was handed to an engine sized at the
        // allocation, so the committed prefill could run past what the engine can hold.
        val engine = LiteRtProvider.engineContextTokens(4096, 32768)
        val committedChars = 500 + 3000 + LiteRtProvider.toolDeclarationCharBudget(engine)
        assertTrue(
            "prefill budget ${committedChars}c must fit the engine's ${engine}t",
            committedChars <= engine * 4,
        )
    }
}


/** Coverage for LiteRT-LM 0.16.1 generation controls that can be tested without JNI. */
class LiteRt0161GenerationControlsTest {

    @Test
    fun `max output token is clamped to engine allocation`() {
        assertEquals(1024, effectiveMaxOutputToken(1024, 4096))
        assertEquals(4096, effectiveMaxOutputToken(8000, 4096))
        assertEquals(null, effectiveMaxOutputToken(null, 4096))
        assertEquals(null, effectiveMaxOutputToken(0, 4096))
    }

    @Test
    fun `thinking follows reasoning level and clamps finite budget`() {
        val off = liteRtThinkingConfig(true, ReasoningLevel.OFF, 4096)!!
        assertFalse(off.enableThinking)
        assertEquals(0, off.thinkingTokenBudget)

        val auto = liteRtThinkingConfig(true, ReasoningLevel.AUTO, 4096)!!
        assertTrue(auto.enableThinking)
        assertEquals(-1, auto.thinkingTokenBudget)

        val high = liteRtThinkingConfig(true, ReasoningLevel.HIGH, 4096)!!
        assertTrue(high.enableThinking)
        assertEquals(4096, high.thinkingTokenBudget)

        assertEquals(null, liteRtThinkingConfig(false, ReasoningLevel.HIGH, 4096))
    }

    @Test
    fun `OpenAI json schema custom body maps to native response format`() {
        val body = CustomBody(
            key = "response_format",
            value = Json.parseToJsonElement(
                """{"type":"json_schema","json_schema":{"schema":{"type":"object","required":["answer"]}}}"""
            ),
        )
        val format = parseLiteRtResponseFormat(listOf(body))!!
        assertEquals(ResponseFormat.Type.JSON_OBJECT, format.type)
        assertTrue(format.schemaOrPattern.contains("\"required\""))
    }

    @Test
    fun `regex custom body maps to native response format`() {
        val body = CustomBody(
            key = "litert_response_regex",
            value = kotlinx.serialization.json.JsonPrimitive("(yes|no)"),
        )
        val format = parseLiteRtResponseFormat(listOf(body))!!
        assertEquals(ResponseFormat.Type.REGEX, format.type)
        assertEquals("(yes|no)", format.schemaOrPattern)
    }

    @Test
    fun `visual budget is enabled only for curated Gemma 4 files`() {
        assertTrue(
            LiteRtModelDefaults.forModelFile("gemma-4-E2B-it.litertlm")
                .supportsVisualTokenBudget
        )
        assertFalse(
            LiteRtModelDefaults.forModelFile("Qwen3_1.7B.litertlm")
                .supportsVisualTokenBudget
        )
    }
}
