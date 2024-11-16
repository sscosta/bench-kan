package pt.isel.har

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Environment
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import pt.isel.har.ui.theme.HARDetectorTheme
import java.io.File
import java.util.*
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var module: Module
    private var predictedActivity by mutableStateOf("")
    private val activityLabels = arrayOf("LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS","WALKING_UPSTAIRS")
    private lateinit var sensorManager: SensorManager
    private var accelerometerData = FloatArray(3)
    private var gyroscopeData = FloatArray(3)
    private val windowData = mutableListOf<FloatArray>()
    private val windowSize = 100
    private val scaler = StandardScaler(6)
    private lateinit var textToSpeech: TextToSpeech
    private val permissionRequestLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            initializeSensors()
        } else {
            // Handle the case where the user denies the permission
        }
    }

    private fun loadLibrary() {
        try {
            val libraryPath = "${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)}/SensorData/libavutil.so"
            System.loadLibrary(libraryPath)
        } catch (e: UnsatisfiedLinkError) {
            e.printStackTrace()
            // Handle the error appropriately
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            HARDetectorTheme {
                SensorDataDisplay(
                    accelerometerData = "Accelerometer Data: ${accelerometerData.joinToString()}",
                    gyroscopeData = "Gyroscope Data: ${gyroscopeData.joinToString()}",
                    predictedActivity = "Predicted Activity: $predictedActivity"
                )
            }
        }

        // Initialize TextToSpeech
        textToSpeech = TextToSpeech(this) { status ->
            if (status != TextToSpeech.ERROR) {
                textToSpeech.language = Locale.US
            }
        }

        // Load your PyTorch module
        val modulePath = "${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)}/SensorData/KAN.pt"
        module = LiteModuleLoader.load(modulePath)

        // Check and request permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) == PackageManager.PERMISSION_GRANTED) {
            initializeSensors()
        } else {
            permissionRequestLauncher.launch(Manifest.permission.BODY_SENSORS)
        }
        fitScalerWithTrainingData()
    }

    private fun initializeSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_NORMAL)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event != null) {
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    accelerometerData = event.values.clone()
                }
                Sensor.TYPE_GYROSCOPE -> {
                    gyroscopeData = event.values.clone()
                }
            }

            val data = accelerometerData + gyroscopeData

            synchronized(this) {
                windowData.add(data)
                if (windowData.size == windowSize) {
                    // Extract features and predict activity
                    val scaledData = scaler.transform(windowData)
                    val features = extractFeatures(scaledData)
                    val predictedLabel = predictActivity(features)
                    println("Predicted Label: $predictedLabel")

                    // Update UI with predicted activity
                    updateUI(predictedLabel)

                    // Clear the windowData
                    windowData.clear()
                }
            }
        }
    }

    private fun fitScalerWithTrainingData() {
        val csvFile = File("${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)}/SensorData/har_mobile_train.csv")
        if (csvFile.exists()) {
            val lines = csvFile.readLines()
            val data = lines.drop(1).map { line ->
                line.split(",").map { it.toFloat() }.toFloatArray()
            }

            val concatenatedData = data.toMutableList()
            //scaler.fit(concatenatedData)
        } else {
            println("Training data file not found.")
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Do something if sensor accuracy changes
    }

    private fun updateUI(activity: String) {
        runOnUiThread {
            predictedActivity = activity
            speak(activity) // Speak the predicted activity
        }
    }

    private fun speak(text: String) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    private fun extractFeatures(segment: MutableList<FloatArray>): FloatArray {
        val numSensors = 6
        val features = FloatArray(numSensors * 4)

        // Print the segment
        println("Segment:")
        segment.forEach { println(it.joinToString()) }

        // Transpose the segment data
        val transposed = Array(numSensors) { FloatArray(segment.size) }
        for (i in segment.indices) {
            for (j in 0 until numSensors) {
                transposed[j][i] = segment[i][j]
            }
        }

        // Print the transposed array
        println("Transposed Array:")
        transposed.forEach { println(it.joinToString()) }

        // Compute statistics for each sensor axis
        for (i in 0 until numSensors) {
            val sensorData = transposed[i]
            val mean = sensorData.average().toFloat()
            val std = sqrt(sensorData.map { (it - mean).pow(2) }.sum() / sensorData.size).toFloat()
            val min = sensorData.minOrNull() ?: 0f
            val max = sensorData.maxOrNull() ?: 0f

            // Print the computed statistics
            println("Sensor $i - Mean: $mean, Std: $std, Min: $min, Max: $max")

            features[i * 4] = mean
            features[i * 4 + 1] = std
            features[i * 4 + 2] = min
            features[i * 4 + 3] = max
        }

        return features
    }

    private fun extractFeatures2(segment: FloatArray): FloatArray {
        val features = FloatArray(24)
        val numSensors = 6
        val windowSize = segment.size / numSensors

        for (i in 0 until numSensors) {
            val sensorData = segment.sliceArray(i * windowSize until (i + 1) * windowSize)
            val mean = sensorData.average().toFloat()
            val std = sqrt(sensorData.map { (it - mean).pow(2) }.sum() / sensorData.size).toFloat()
            val min = sensorData.minOrNull() ?: 0f
            val max = sensorData.maxOrNull() ?: 0f

            features[i * 4] = mean
            features[i * 4 + 1] = std
            features[i * 4 + 2] = min
            features[i * 4 + 3] = max
        }

        return features
    }

    private fun predictActivity(data: FloatArray): String {
        // Create input tensor from the raw data
        val inputTensor = Tensor.fromBlob(data, longArrayOf(1, data.size.toLong()))

        // Perform inference with the module
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val outputData = outputTensor.dataAsFloatArray

        // Determine the predicted activity label
        var maxIndex = 0
        var maxValue = outputData[0]

        for (i in 1 until outputData.size) {
            if (outputData[i] > maxValue) {
                maxValue = outputData[i]
                maxIndex = i
            }
        }

        return activityLabels[maxIndex]
    }

    class StandardScaler(private val numFeatures: Int) {
        //private val mean = FloatArray(numFeatures)
        //private val std = FloatArray(numFeatures)
        //private val mean = floatArrayOf(-5.25738633e+00f, 1.35860153e+00f, 5.58978456e+00f, 8.12027269e-03f, 4.38624162e-03f, 2.79569972e-04f)
        private val mean = floatArrayOf(-5.25738633e+00f,  1.35860153e+00f,  5.58978456e+00f,  8.12027269e-03f,
            4.38624162e-03f,  2.79569972e-04f)
        //private val std = floatArrayOf(4.32453707f, 3.50257806f, 3.32556351f, 0.33050506f, 0.70318933f, 0.23725164f)
        private val std = floatArrayOf(4.32453707f, 3.50257806f, 3.32556351f, 0.33050506f, 0.70318933f, 0.23725164f)

        fun MutableList<FloatArray>.concatenate(): FloatArray {
            val size = this.sumOf { it.size }
            val concatenatedArray = FloatArray(size)
            var currentIndex = 0
            for (array in this) {
                System.arraycopy(array, 0, concatenatedArray, currentIndex, array.size)
                currentIndex += array.size
            }
            return concatenatedArray
        }

        fun fit(data: MutableList<FloatArray>) {
            val size = data.size
            val flattenedData = data.concatenate()

            for (i in 0 until numFeatures) {
                val featureData = flattenedData.sliceArray(i * size until (i + 1) * size)
                mean[i] = featureData.average().toFloat()
                std[i] = sqrt(featureData.map { (it - mean[i]).pow(2) }.sum() / featureData.size).toFloat()
            }
        }

        fun transform(data: MutableList<FloatArray>): MutableList<FloatArray> {
            val transformedData = mutableListOf<FloatArray>()
            val size = data.size

            for (row in data) {
                val transformedRow = FloatArray(numFeatures)
                for (i in 0 until numFeatures) {
                    transformedRow[i] = (row[i] - mean[i]) / std[i]
                }
                transformedData.add(transformedRow)
            }
            return transformedData
        }

        fun fitTransform(data: MutableList<FloatArray>): MutableList<FloatArray> {
            fit(data)
            return transform(data)
        }
    }

    @Preview(showBackground = true)
    @Composable
    fun SensorDataDisplayPreview() {
        HARDetectorTheme {
            SensorDataDisplay(
                accelerometerData = "X: 0.0, Y: 0.0, Z: 0.0",
                gyroscopeData = "X: 0.0, Y: 0.0, Z: 0.0",
                predictedActivity = "Predicted Activity: Walking"
            )
        }
    }

    @Composable
    fun SensorDataDisplay(
        accelerometerData: String,
        gyroscopeData: String,
        predictedActivity: String,
        modifier: Modifier = Modifier
    ) {
        Column(modifier = modifier) {
            Text(
                text = accelerometerData,
                modifier = Modifier.padding(8.dp)
            )
            Text(
                text = gyroscopeData,
                modifier = Modifier.padding(8.dp)
            )
            Text(
                text = predictedActivity, // Display predicted activity
                modifier = Modifier.padding(8.dp)
            )
        }
    }

    override fun onDestroy() {
        // Shutdown TTS when activity is destroyed
        if (textToSpeech != null) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        super.onDestroy()
    }
}
