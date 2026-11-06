import android.content.Context;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class YoloPip {

    private final Interpreter intrepreter;

    // Java has to decode the output of the model from tensors
    public YoloPip(Context context) {
        ByteBuffer buffer = loadModel(context, "best.tflite");
        intrepreter = new Interpreter(buffer);

        int[] inputShape = intrepreter.getInputTensor(0).shape();
    }

    private static ByteBuffer loadModel(Context context, String flName) {
        // Not done yet, figure out how to load the tflite file using FileInputStream
        FileInputStream input = new FileInputStream();
    }

}
