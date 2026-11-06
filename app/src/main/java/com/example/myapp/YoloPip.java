// AI has been used to debug this

package com.example.myapp;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class YoloPip {

    private static final String TAG = "YoloPip";
    private static final String MODEL_FILE = "best.tflite";

    private final Interpreter interpreter;
    private final int[] inputShape;

    public YoloPip(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options().setNumThreads(4);
        ByteBuffer buffer = loadModel(context, MODEL_FILE);
        interpreter = new Interpreter(buffer, options);

        inputShape = interpreter.getInputTensor(0).shape();
        Log.i(TAG, "Model input shape: " + java.util.Arrays.toString(inputShape)
                + ", output shape: " + java.util.Arrays.toString(interpreter.getOutputTensor(0).shape()));
    }

    private static ByteBuffer loadModel(Context context, String fileName) throws IOException {
        try (AssetFileDescriptor fd = context.getAssets().openFd(fileName)) {
            try (FileInputStream stream = new FileInputStream(fd.getFileDescriptor())) {
                FileChannel channel = stream.getChannel();
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getLength());
            }
        }
    }

    // Allocate memory for storing each value in a tensorshape
    private static ByteBuffer allocateFor(int[] tensorShape) {
        int elementCount = 1;
        for (int dim : tensorShape) {
            elementCount *= dim;
        }
        return ByteBuffer.allocateDirect(elementCount * Float.BYTES)
                .order(ByteOrder.nativeOrder());
    }

    public ByteBuffer createInputBuffer() {
        return allocateFor(interpreter.getInputTensor(0).shape());
    }

    public ByteBuffer createOutputBuffer() {
        return allocateFor(interpreter.getOutputTensor(0).shape());
    }

    public void run(ByteBuffer input, ByteBuffer output) {
        // IMPORATANT REUSE BUFFERS INSTEAD OF ALLOCATING EVERYTIME -> performace optimization
        input.rewind();
        output.rewind();
        interpreter.run(input, output);
        output.rewind();
    }

    public int[] getInputShape() {
        return inputShape;
    }

    public int getModelWidth() {
        int[] s = inputShape;
        // NCHW: [1, 3, H, W]  vs  NHWC: [1, H, W, 3]
        if (s.length >= 4 && s[1] == 3) return s[3];
        return s.length >= 4 ? s[2] : s[0];
    }

    public int getModelHeight() {
        int[] s = inputShape;
        if (s.length >= 4 && s[1] == 3) return s[2];
        return s.length >= 4 ? s[1] : s[0];
    }

    public int[] getOutputShape() {
        return interpreter.getOutputTensor(0).shape();
    }

    public void destroy() {
        interpreter.close();
    }
}
