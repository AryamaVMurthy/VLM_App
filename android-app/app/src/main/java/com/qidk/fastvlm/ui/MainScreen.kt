package com.qidk.fastvlm.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun MainScreen(viewModel: MainViewModel) {
  val state by viewModel.uiState.collectAsStateWithLifecycle()
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current

  var showTranscript by remember { mutableStateOf(false) }
  var cameraPermissionGranted by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED,
    )
  }
  var audioPermissionGranted by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
        PackageManager.PERMISSION_GRANTED,
    )
  }

  val cameraPermissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      cameraPermissionGranted = granted
    }
  val audioPermissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      audioPermissionGranted = granted
    }

  LaunchedEffect(Unit) {
    viewModel.initializeIfNeeded()
    if (!cameraPermissionGranted) {
      cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }
    if (!audioPermissionGranted) {
      audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }
  }

  LaunchedEffect(state.activeRequestId, state.isRecordingVoice, state.isTranscribingVoice) {
    if (state.isRecordingVoice || state.isTranscribingVoice) {
      showTranscript = false
    }
  }

  val holdToSpeakEnabled =
    cameraPermissionGranted &&
      audioPermissionGranted &&
      state.cameraReady &&
      state.speechReady &&
      !state.isInitializing &&
      !state.isTranscribingVoice

  val statusText =
    when {
      state.isInitializing -> "Initializing"
      !cameraPermissionGranted -> "Camera permission required"
      !audioPermissionGranted -> "Microphone permission required"
      !state.cameraReady -> "Preparing camera"
      state.isRecordingVoice -> "Listening"
      state.isTranscribingVoice -> "Transcribing"
      state.transitionState == "preparing_image" -> "Preparing frame"
      state.transitionState == "thinking" -> "Thinking"
      state.transitionState == "answering_stream" -> "Answering"
      state.isSpeaking || state.transitionState == "speaking_live" -> "Speaking"
      state.errorMessage != null -> "Action needed"
      else -> "Ready"
    }

  Box(modifier = Modifier.fillMaxSize()) {
    if (cameraPermissionGranted) {
      CameraPreview(
        modifier = Modifier.fillMaxSize(),
        onPreviewReady = { previewView ->
          viewModel.bindCamera(lifecycleOwner, previewView)
        },
      )
    } else {
      Box(
        modifier = Modifier.fillMaxSize().background(Color(0xFF070B12)),
        contentAlignment = Alignment.Center,
      ) {
        Button(onClick = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) }) {
          Text("Grant Camera")
        }
      }
    }

    Box(
      modifier =
        Modifier.fillMaxSize().background(
          Brush.verticalGradient(
            colors =
              listOf(
                Color(0x99000000),
                Color(0x1A000000),
                Color(0x59000000),
              ),
          ),
        ),
    )

    Column(
      modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 18.dp),
      verticalArrangement = Arrangement.SpaceBetween,
    ) {
      Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        color = Color(0x8A0A0F16),
        tonalElevation = 0.dp,
      ) {
        Row(
          modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Text(
            text = statusText,
            style = MaterialTheme.typography.labelMedium,
            color = Color.White,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
          )

          Surface(
            shape = RoundedCornerShape(999.dp),
            color = Color(0x2EFFFFFF),
            modifier = Modifier.clickable { showTranscript = !showTranscript },
          ) {
            Text(
              text = if (showTranscript) "Hide Text" else "Show Text",
              modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
              style = MaterialTheme.typography.labelSmall,
              color = Color.White,
            )
          }
        }
      }

      Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
      ) {
        state.errorMessage?.let { message ->
          Surface(
            shape = RoundedCornerShape(999.dp),
            color = Color(0xD9321C1C),
          ) {
            Text(
              text = message,
              modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
              style = MaterialTheme.typography.labelSmall,
              color = Color.White,
              maxLines = 1,
              overflow = TextOverflow.Ellipsis,
            )
          }
        }

        if (!audioPermissionGranted) {
          Button(onClick = { audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO) }) {
            Text("Grant Microphone")
          }
        }

        Surface(
          modifier =
            Modifier
              .size(108.dp)
              .pointerInput(holdToSpeakEnabled) {
                if (!holdToSpeakEnabled) return@pointerInput
                detectTapGestures(
                  onPress = {
                    viewModel.startVoiceRecording()
                    tryAwaitRelease()
                    viewModel.stopVoiceRecordingAndAsk()
                  },
                )
              },
          shape = CircleShape,
          color =
            when {
              state.isRecordingVoice -> Color(0xFFFF5B5B)
              state.isRunning || state.isTranscribingVoice -> Color(0xFF3B82F6)
              holdToSpeakEnabled -> Color(0xE8FFFFFF)
              else -> Color(0x66FFFFFF)
            },
          tonalElevation = 6.dp,
        ) {
          Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
              text =
                when {
                  state.isRecordingVoice -> "Release"
                  state.isTranscribingVoice -> "Wait"
                  state.isRunning -> "..."
                  else -> "Hold"
                },
              style = MaterialTheme.typography.titleMedium,
              color = if (state.isRecordingVoice || state.isRunning || state.isTranscribingVoice) Color.White else Color(0xFF101828),
            )
          }
        }

        Text(
          text =
            when {
              !holdToSpeakEnabled && cameraPermissionGranted && audioPermissionGranted -> "Voice assistant warming up"
              !holdToSpeakEnabled -> "Press and hold after permissions are granted"
              state.isRecordingVoice -> "Release to ask"
              else -> "Hold to speak"
            },
          style = MaterialTheme.typography.bodySmall,
          color = Color.White.copy(alpha = 0.92f),
        )

        if ((state.isRecordingVoice || state.isTranscribingVoice) && state.question.isNotBlank()) {
          Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            color = Color(0x660A0F16),
          ) {
            Text(
              text = "Heard: ${state.question}",
              modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
              style = MaterialTheme.typography.labelSmall,
              color = Color.White.copy(alpha = 0.95f),
              maxLines = 2,
              overflow = TextOverflow.Ellipsis,
            )
          }
        }
      }
    }

    if (showTranscript) {
      Surface(
        modifier =
          Modifier
            .align(Alignment.BottomCenter)
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 24.dp)
            .heightIn(max = 300.dp),
        shape = RoundedCornerShape(20.dp),
        color = Color(0xE610172A),
        tonalElevation = 8.dp,
      ) {
        Column(
          modifier = Modifier.fillMaxWidth().padding(14.dp),
          verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(
              text = "Assistant Text",
              style = MaterialTheme.typography.titleSmall,
              color = Color.White,
            )
            Surface(
              shape = RoundedCornerShape(999.dp),
              color = Color(0x33FFFFFF),
              modifier = Modifier.clickable { showTranscript = false },
            ) {
              Text(
                text = "Close",
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
              )
            }
          }
          Text(
            text =
              if (state.answer.isBlank()) {
                "No response text yet."
              } else {
                state.answer
              },
            modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White.copy(alpha = 0.96f),
          )
        }
      }
    }
  }
}

@Composable
private fun CameraPreview(
  modifier: Modifier,
  onPreviewReady: (PreviewView) -> Unit,
) {
  AndroidView(
    modifier = modifier,
    factory = { context ->
      PreviewView(context).apply {
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        scaleType = PreviewView.ScaleType.FILL_CENTER
        // SM8750P board camera stream is mounted inverted; compensate in preview.
        rotation = 180f
        onPreviewReady(this)
      }
    },
  )
}
