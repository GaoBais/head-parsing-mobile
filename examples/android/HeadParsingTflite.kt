package com.example.headparsing

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class HeadParsingTflite(
    context: Context,
    assetName: String = "head_parsing_mobile_320_teeth_fp16.tflite",
    numThreads: Int = 4,
    delegate: Delegate? = null,
) : Closeable {
    private val model: MappedByteBuffer = loadModel(context, assetName)
    private val options = Interpreter.Options().apply {
        setNumThreads(numThreads)
        delegate?.let { addDelegate(it) }
    }
    private val interpreter = Interpreter(model, options)

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(BATCH * CHANNELS * SIZE * SIZE * FLOAT_BYTES).order(ByteOrder.nativeOrder())
    private val outputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(BATCH * NUM_CLASSES * SIZE * SIZE * FLOAT_BYTES).order(ByteOrder.nativeOrder())

    fun segment(bitmap: Bitmap): SegmentationResult {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        writeNchwInput(resized)

        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        val mask = ByteArray(SIZE * SIZE)
        val histogram = IntArray(NUM_CLASSES)
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                var bestClass = 0
                var bestLogit = Float.NEGATIVE_INFINITY
                for (classId in 0 until NUM_CLASSES) {
                    val value = getLogit(classId, y, x)
                    if (value > bestLogit) {
                        bestLogit = value
                        bestClass = classId
                    }
                }
                val index = y * SIZE + x
                mask[index] = bestClass.toByte()
                histogram[bestClass] += 1
            }
        }
        return SegmentationResult(mask = mask, width = SIZE, height = SIZE, histogram = histogram)
    }

    fun inputShape(): IntArray = interpreter.getInputTensor(0).shape()

    fun outputShape(): IntArray = interpreter.getOutputTensor(0).shape()

    override fun close() {
        interpreter.close()
    }

    private fun writeNchwInput(bitmap: Bitmap) {
        val pixels = IntArray(SIZE * SIZE)
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        inputBuffer.rewind()

        for (channel in 0 until CHANNELS) {
            for (y in 0 until SIZE) {
                for (x in 0 until SIZE) {
                    val color = pixels[y * SIZE + x]
                    val raw = when (channel) {
                        0 -> (color shr 16) and 0xFF
                        1 -> (color shr 8) and 0xFF
                        else -> color and 0xFF
                    }
                    val normalized = (raw / 255.0f - MEAN[channel]) / STD[channel]
                    inputBuffer.putFloat(normalized)
                }
            }
        }
        inputBuffer.rewind()
    }

    private fun getLogit(classId: Int, y: Int, x: Int): Float {
        val floatIndex = ((classId * SIZE + y) * SIZE + x)
        return outputBuffer.getFloat(floatIndex * FLOAT_BYTES)
    }

    private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }

    data class SegmentationResult(
        val mask: ByteArray,
        val width: Int,
        val height: Int,
        val histogram: IntArray,
    )

    companion object {
        const val SIZE = 320
        const val NUM_CLASSES = 20
        const val TEETH_CLASS_ID = 19

        private const val BATCH = 1
        private const val CHANNELS = 3
        private const val FLOAT_BYTES = 4
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
