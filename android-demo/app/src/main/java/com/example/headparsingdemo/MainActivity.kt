package com.example.headparsingdemo

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.Locale
import kotlin.math.min

class MainActivity : Activity() {
    private lateinit var statusView: TextView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        val runButton = Button(this).apply {
            text = "Run TFLite Smoke Test"
            setOnClickListener { runSmokeTest() }
        }
        statusView = TextView(this).apply {
            text = "Copy model and sample_107.jpg into app/src/main/assets, then run."
            textSize = 14f
        }
        imageView = ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
        }

        root.addView(runButton, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        root.addView(statusView, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        root.addView(
            imageView,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f).apply {
                gravity = Gravity.CENTER
                topMargin = 24
            },
        )

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun runSmokeTest() {
        try {
            val input = loadBitmapFromAssets(SAMPLE_ASSET)
            HeadParsingTflite(this).use { model ->
                val start = SystemClock.elapsedRealtimeNanos()
                val result = model.segment(input)
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                imageView.setImageBitmap(colorizeMask(result.mask, result.width, result.height))

                statusView.text = buildString {
                    appendLine("runtime: Android LiteRT/TFLite CPU")
                    appendLine("inputShape: ${model.inputShape().contentToString()}")
                    appendLine("outputShape: ${model.outputShape().contentToString()}")
                    appendLine("latencyMs: ${String.format(Locale.US, "%.2f", elapsedMs)}")
                    appendLine("teeth pixels: ${result.histogram[HeadParsingTflite.TEETH_CLASS_ID]}")
                    appendLine()
                    appendLine("top classes:")
                    appendLine(topClasses(result.histogram))
                }
            }
        } catch (error: Throwable) {
            statusView.text = "Smoke test failed:\n${error.stackTraceToString()}"
        }
    }

    private fun loadBitmapFromAssets(name: String): Bitmap {
        assets.open(name).use { input ->
            return requireNotNull(BitmapFactory.decodeStream(input)) {
                "Could not decode asset image: $name"
            }
        }
    }

    private fun colorizeMask(mask: ByteArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)
        for (index in mask.indices) {
            val classId = mask[index].toInt() and 0xFF
            val color = PALETTE[min(classId, PALETTE.lastIndex)]
            pixels[index] = (0xFF shl 24) or (color[0] shl 16) or (color[1] shl 8) or color[2]
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun topClasses(histogram: IntArray): String {
        return histogram
            .mapIndexed { index, count -> index to count }
            .filter { it.second > 0 }
            .sortedByDescending { it.second }
            .take(6)
            .joinToString(separator = "\n") { (index, count) ->
                "${CLASS_NAMES[index]}: $count"
            }
    }

    companion object {
        private const val SAMPLE_ASSET = "sample_107.jpg"

        private val CLASS_NAMES = arrayOf(
            "background",
            "skin",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hair",
            "hat",
            "teeth",
        )

        private val PALETTE = arrayOf(
            intArrayOf(0, 0, 0),
            intArrayOf(255, 85, 0),
            intArrayOf(255, 170, 0),
            intArrayOf(255, 0, 85),
            intArrayOf(255, 0, 170),
            intArrayOf(0, 255, 0),
            intArrayOf(85, 255, 0),
            intArrayOf(170, 255, 0),
            intArrayOf(0, 255, 85),
            intArrayOf(0, 255, 170),
            intArrayOf(0, 0, 255),
            intArrayOf(85, 0, 255),
            intArrayOf(170, 0, 255),
            intArrayOf(0, 85, 255),
            intArrayOf(0, 170, 255),
            intArrayOf(255, 255, 0),
            intArrayOf(255, 255, 85),
            intArrayOf(255, 255, 170),
            intArrayOf(255, 0, 255),
            intArrayOf(0, 255, 255),
        )
    }
}
