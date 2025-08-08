import AVFoundation
import MLX

public actor KokoroTTSService {
    private let kokoroTTSEngine: KokoroTTS
    private let audioEngine: AVAudioEngine
    private let playerNode: AVAudioPlayerNode
    private let audioFormat: AVAudioFormat

    // Simple state tracking
    private var isInitialized = false
    
    public init(modelFilePath: String) throws {
        kokoroTTSEngine = KokoroTTS(modelFilePath: modelFilePath)
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(KokoroTTS.Constants.sampleRate), channels: 1) else {
            throw SpeechServiceError.audioFormatCreationFailed
        }
        audioFormat = format
        
        try setupAudioSystem()
    }
    
    deinit {
        cleanupAudioSystem()
    }
    
    public func say(_ text: String, _ voice: TTSVoice, speed: Float = 1.0) async throws {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }
        
        // Generate audio and schedule buffers directly to player node
        try await generateAndScheduleAudio(text: trimmedText, voice: voice, speed: speed)
    }
    
    public func stop() {
        stopAndClearPlayback()
    }
    
    // MARK: - Private Methods
    
    private func setupAudioSystem() throws {
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)
        
        try audioEngine.start()
        isInitialized = true
    }
    
    private func cleanupAudioSystem() {
        stopAndClearPlayback()
        
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        
        isInitialized = false
    }
    
    private func stopAndClearPlayback() {
        // Stop current playback and clear the internal queue
        if playerNode.isPlaying {
            playerNode.stop()
        }
    }
    
    private func generateAndScheduleAudio(text: String, voice: TTSVoice, speed: Float) async throws {
        // Ensure audio engine is running
        if !audioEngine.isRunning {
            try audioEngine.start()
        }
        
        // Generate audio stream and schedule buffers directly
        let audioStream = kokoroTTSEngine.generateAudioStream(voice: voice, text: text, speed: speed)
        
        for try await audioBuffer in audioStream {
            if let pcmBuffer = convertToPCMBuffer(audioBuffer) {
                // Schedule buffer to the player node's internal queue
                playerNode.scheduleBuffer(pcmBuffer, completionHandler: nil)

                if !playerNode.isPlaying {
                    playerNode.play()
                }
            }
        }
    }
    
    private func convertToPCMBuffer(_ audioBuffer: MLXArray) -> AVAudioPCMBuffer? {
        let audioShape = audioBuffer.shape
        
        // Skip empty chunks
        guard !isAudioEmpty(shape: audioShape) else {
            return nil
        }
        
        // Extract audio data
        let (frameCount, audioData) = extractAudioData(from: audioBuffer)
        guard frameCount > 0 && !audioData.isEmpty else {
            return nil
        }
        
        // Create PCM buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return nil
        }
        
        buffer.frameLength = buffer.frameCapacity
        
        // Copy data with volume boost
        let channels = buffer.floatChannelData!
        for i in 0..<min(frameCount, audioData.count, Int(buffer.frameCapacity)) {
            // Apply volume boost (25%) with clipping prevention
            channels[0][i] = min(max(audioData[i] * 1.25, -0.98), 0.98)
        }
        
        return buffer
    }
    
    private func isAudioEmpty(shape: [Int]) -> Bool {
        if shape.count == 1 {
            return shape[0] <= 1
        } else if shape.count == 2 {
            return shape[1] <= 1
        }
        return true
    }
    
    private func extractAudioData(from audioBuffer: MLXArray) -> (frameCount: Int, audioData: [Float]) {
        let audioShape = audioBuffer.shape
        
        if audioShape.count == 1 {
            // 1D array [samples]
            let frameCount = audioShape[0]
            return (frameCount, audioBuffer.asArray(Float.self))
        } else if audioShape.count == 2 {
            // 2D array [1, samples]
            let frameCount = audioShape[1]
            let firstBatch = audioBuffer[0]
            return (frameCount, firstBatch.asArray(Float.self))
        }
        
        return (0, [])
    }
}

// MARK: - Error Types

public enum SpeechServiceError: Error, LocalizedError {
    case audioFormatCreationFailed
    case playbackInterrupted
    case playbackTimeout
    case audioEngineStartFailed
    
    public var errorDescription: String? {
        switch self {
        case .audioFormatCreationFailed:
            return "Failed to create audio format"
        case .playbackInterrupted:
            return "Playback was interrupted"
        case .playbackTimeout:
            return "Playback timed out"
        case .audioEngineStartFailed:
            return "Failed to start audio engine"
        }
    }
} 
