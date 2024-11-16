package pt.isel.collectdata

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import pt.isel.collectdata.ui.theme.CollectDataTheme
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private val sensorData = mutableListOf<String>()
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault())
    private var textToSpeech: TextToSpeech? = null

    private var accelValues by mutableStateOf(floatArrayOf(0f, 0f, 0f))
    private var gyroValues by mutableStateOf(floatArrayOf(0f, 0f, 0f))
    private var isCollectingData by mutableStateOf(false)
    private var currentActivityLabel by mutableStateOf("")
    private var subjectName by mutableStateOf("")
    private var selectedActivity by mutableStateOf("LAYING")

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale.US
            }
        }

        if (allPermissionsGranted()) {
            // Permissions are already granted, proceed with the rest of your app logic
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ), REQUEST_CODE_PERMISSIONS)
        }

        setContent {
            CollectDataTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding).padding(16.dp)) {
                        TextField(
                            value = subjectName,
                            onValueChange = { subjectName = it },
                            label = { Text("Enter Subject Name") },
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        SensorDataDisplay(
                            accelValues = accelValues,
                            gyroValues = gyroValues,
                            currentActivityLabel = currentActivityLabel
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        ActivitySelection(
                            selectedActivity = selectedActivity,
                            onActivitySelected = { selectedActivity = it }
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        ActivityButtons(
                            subjectName = subjectName,
                            selectedActivity = selectedActivity,
                            isCollectingData = isCollectingData,
                            startDataCollection = { activity -> startDataCollection(activity) },
                            stopDataCollection = { stopDataCollection() },
                            startActivityProtocol = { startActivityProtocol() }
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Activity Protocol Order:")
                        Text("LAYING\nSITTING\nSTANDING\nWALKING\nWALKING_UPSTAIRS\nWALKING_DOWNSTAIRS")
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted(): Boolean {
        val permissions = arrayOf(
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
        return permissions.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isCollectingData) return

        event?.let {
            val timestamp = dateFormat.format(Date())
            when (it.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    accelValues = it.values.clone()
                    val accelData =
                        "\"${timestamp}\",${it.values[0]},${it.values[1]},${it.values[2]},,,,\"${currentActivityLabel}\",\"$subjectName\""
                    sensorData.add(accelData)
                }

                Sensor.TYPE_GYROSCOPE -> {
                    gyroValues = it.values.clone()
                    val gyroData = "\"${timestamp}\",,,,${it.values[0]},${it.values[1]},${it.values[2]},\"${currentActivityLabel}\",\"$subjectName\""
                    sensorData.add(gyroData)
                }

                else -> {
                    // Handle other sensor types or ignore
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Do nothing
    }

    private fun startDataCollection(activityLabel: String) {
        currentActivityLabel = activityLabel
        isCollectingData = true
        sensorData.clear()

        accelerometer?.also { acc ->
            sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.also { gyro ->
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST)
        }

        textToSpeech?.speak(activityLabel, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    private fun stopDataCollection() {
        isCollectingData = false
        sensorManager.unregisterListener(this)
        saveDataToCsv()
    }

    private fun saveDataToCsv() {
        val directory = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "SensorData"
        )
        if (!directory.exists()) {
            directory.mkdirs()
        }
        val file = File(directory, "${subjectName}_${currentActivityLabel}_${dateFormat.format(Date())}.csv")
        try {
            val fileWriter = FileWriter(file)
            fileWriter.append("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,activity,subject\n")
            for (data in sensorData) {
                fileWriter.append("$data\n")
            }
            fileWriter.flush()
            fileWriter.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

/**    private fun startActivityProtocol() {
        coroutineScope.launch {
            val activities = listOf("LAYING", "SITTING", "STANDING", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS")
            for (activity in activities) {
                startDataCollection(activity)
                delay(30000) // Collect data for 30 seconds
                stopDataCollection()
                delay(5000) // Pause for 5 seconds
            }
        }
    }**/
@RequiresApi(Build.VERSION_CODES.LOLLIPOP)
private fun startActivityProtocol() {
    coroutineScope.launch {
        val activities = listOf("LAYING", "SITTING", "STANDING", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS")
        for (activity in activities) {
            startDataCollection(activity)
            delay(25000L)
            // Countdown logic
            for (i in 5 downTo 1) {
                textToSpeech?.speak(i.toString(), TextToSpeech.QUEUE_FLUSH, null, null)
                delay(1000)
            }
            textToSpeech?.speak("STOP",TextToSpeech.QUEUE_FLUSH, null, null)
            stopDataCollection()
            delay(5000) // Pause for 5 seconds before the next activity
        }
    }
}

    @Composable
    fun ActivitySelection(
        selectedActivity: String,
        onActivitySelected: (String) -> Unit
    ) {
        val activities = listOf(
            "LAYING",
            "SITTING",
            "STANDING",
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS"
        )

        // State to control the dropdown expanded state
        var expanded by remember { mutableStateOf(false) }

        Box(modifier = Modifier.fillMaxWidth()) {
            TextButton(onClick = { expanded = !expanded }) {
                Text(selectedActivity) // Display the currently selected activity
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                activities.forEach { activity ->
                    DropdownMenuItem(
                        text = { Text(activity) },
                        onClick = {
                            onActivitySelected(activity)
                            expanded = false
                        }
                    )
                }
            }
        }
    }

    @Composable
    fun ActivityButtons(
        subjectName: String,
        selectedActivity: String,
        isCollectingData: Boolean,
        startDataCollection: (String) -> Unit,
        stopDataCollection: () -> Unit,
        startActivityProtocol: () -> Unit
    ) {
        Column {
            Button(onClick = {
                if (subjectName.isNotBlank()) {
                    if (isCollectingData) stopDataCollection()
                    else startDataCollection(selectedActivity)
                }
            }) {
                Text(if (isCollectingData) "Stop and Save Data" else "Start $selectedActivity")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = { if (subjectName.isNotBlank()) startActivityProtocol() }) {
                Text("Start Activity Protocol")
            }
        }
    }

    override fun onDestroy() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        super.onDestroy()
    }
}

@Composable
fun SensorDataDisplay(accelValues: FloatArray, gyroValues: FloatArray, currentActivityLabel: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text(text = "Current Activity: $currentActivityLabel")
        Text(text = "Accelerometer Data: x=${accelValues[0]}, y=${accelValues[1]}, z=${accelValues[2]}")
        Text(text = "Gyroscope Data: x=${gyroValues[0]}, y=${gyroValues[1]}, z=${gyroValues[2]}")
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    CollectDataTheme {
        Column {
            TextField(
                value = "",
                onValueChange = {},
                label = { Text("Enter Subject Name") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
            SensorDataDisplay(accelValues = floatArrayOf(0f, 0f, 0f), gyroValues = floatArrayOf(0f, 0f, 0f), currentActivityLabel = "LAYING")
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = { }) {
                Text(text = "Start Activity Protocol")
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text("Activity Protocol Order:")
            Text("LAYING\nSITTING\nSTANDING\nWALKING\nWALKING_UPSTAIRS\nWALKING_DOWNSTAIRS")
        }
    }
}
