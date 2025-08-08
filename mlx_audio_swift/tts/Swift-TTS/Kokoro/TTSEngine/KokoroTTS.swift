//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Available voices
public enum TTSVoice: String, CaseIterable, Sendable {
  case afAlloy
  case afAoede
  case afBella
  case afHeart
  case afJessica
  case afKore
  case afNicole
  case afNova
  case afRiver
  case afSarah
  case afSky
  case amAdam
  case amEcho
  case amEric
  case amFenrir
  case amLiam
  case amMichael
  case amOnyx
  case amPuck
  case amSanta
  case bfAlice
  case bfEmma
  case bfIsabella
  case bfLily
  case bmDaniel
  case bmFable
  case bmGeorge
  case bmLewis
  case efDora
  case emAlex
  case ffSiwis
  case hfAlpha
  case hfBeta
  case hfOmega
  case hmPsi
  case ifSara
  case imNicola
  case jfAlpha
  case jfGongitsune
  case jfNezumi
  case jfTebukuro
  case jmKumo
  case pfDora
  case pmSanta
  case zfZiaobei
  case zfXiaoni
  case zfXiaoxiao
  case zfZiaoyi
  case zmYunjian
  case zmYunxi
  case zmYunxia
  case zmYunyang
}

// Main class, encapsulates the whole Kokoro text-to-speech pipeline
public class KokoroTTS {
  enum KokoroTTSError: Error {
    case tooManyTokens
    case sentenceSplitError
    case modelNotInitialized
  }

  private var bert: CustomAlbert!
  private var bertEncoder: Linear!
  private var durationEncoder: DurationEncoder!
  private var predictorLSTM: LSTM!
  private var durationProj: Linear!
  private var prosodyPredictor: ProsodyPredictor!
  private var textEncoder: TextEncoder!
  private var decoder: Decoder!
  private var eSpeakEngine: ESpeakNGEngine!
  private var kokoroTokenizer: KokoroTokenizer!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!
    private let modelFilePath: String

  // Flag to indicate if model components are initialized
  private var isModelInitialized = false

  // Callback type for streaming audio generation
  public typealias AudioChunkCallback = (MLXArray) -> Void

    init(modelFilePath: String) {
        self.modelFilePath = modelFilePath
      compile(enable: true)
      
      // Only set memory limits on devices with less than 5GB of RAM
      let physicalMemory = ProcessInfo.processInfo.physicalMemory
      let fiveGB: UInt64 = 5 * 1024 * 1024 * 1024
      
      if physicalMemory < fiveGB {
          MLX.GPU.set(memoryLimit: 512 * 1024 * 1024, relaxed: false)
          MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
          print("Applied MLX memory limits for device with \(physicalMemory / 1024 / 1024) MB RAM")
      } else {
          MLX.GPU.set(memoryLimit: 2048 * 1024 * 1024, relaxed: false)
          MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)
      }
  }

  // Reset the model to free up memory
  public func resetModel(preserveTextProcessing: Bool = true) {
    // Reset heavy ML model components
    bert = nil
    bertEncoder = nil
    durationEncoder = nil
    predictorLSTM = nil
    durationProj = nil
    prosodyPredictor = nil
    textEncoder = nil
    decoder = nil
    voice = nil
    chosenVoice = nil
    isModelInitialized = false

    // Optionally preserve text processing components for faster restart
    if !preserveTextProcessing {
      if let _ = eSpeakEngine {
        // Ensure eSpeakEngine is terminated properly
        eSpeakEngine = nil
      }
      kokoroTokenizer = nil
    }

    // Use plain autoreleasepool to encourage memory release
    autoreleasepool { }
  }

  // Initialize model on demand
  private func ensureModelInitialized() throws {
    if isModelInitialized {
      return
    }

    // Initialize text processing components first (less expensive)
    if eSpeakEngine == nil {
      eSpeakEngine = try ESpeakNGEngine()
    }

    if kokoroTokenizer == nil {
      kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)
    }

    autoreleasepool {
        let sanitizedWeights = KokoroWeightLoader.loadWeights(filePath: modelFilePath)

      bert = CustomAlbert(weights: sanitizedWeights, config: AlbertModelArgs())
      bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
      durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: 512, styDim: 128, nlayers: 6)

      predictorLSTM = LSTM(
        inputSize: 512 + 128,
        hiddenSize: 512 / 2,
        wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
        whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
        biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
        biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
        wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
        whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
        biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
        biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
      )

      durationProj = Linear(
        weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
        bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
      )

      prosodyPredictor = ProsodyPredictor(
        weights: sanitizedWeights,
        styleDim: 128,
        dHid: 512
      )

      textEncoder = TextEncoder(
        weights: sanitizedWeights,
        channels: 512,
        kernelSize: 5,
        depth: 3,
        nSymbols: 178
      )

      decoder = Decoder(
        weights: sanitizedWeights,
        dimIn: 512,
        styleDim: 128,
        dimOut: 80,
        resblockKernelSizes: [3, 7, 11],
        upsampleRates: [10, 6],
        upsampleInitialChannel: 512,
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsampleKernelSizes: [20, 12],
        genIstftNFft: 20,
        genIstftHopSize: 5
      )
    }

    isModelInitialized = true
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    speed: Float
  ) throws -> MLXArray {
    // Time the token processing
    BenchmarkTimer.shared.create(id: "generateAudioForTokens", parent: "audioGeneration")
    defer { BenchmarkTimer.shared.stop(id: "generateAudioForTokens") }
    
    // Create a fresh autorelease pool for the entire process
    return try autoreleasepool { () -> MLXArray in
      // Start with the standard processing
      try autoreleasepool {
        let paddedInputIdsBase = [0] + inputIds + [0]
        let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])

        let inputLengths = MLXArray(paddedInputIds.dim(-1))

        let inputLengthMax: Int = MLX.max(inputLengths).item()
        var textMask = MLXArray(0 ..< inputLengthMax)

        textMask = textMask + 1 .> inputLengths

        textMask = textMask.expandedDimensions(axes: [0])

        let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
        let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
        let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)

        return try autoreleasepool { () -> MLXArray in
          // Time BERT processing
          BenchmarkTimer.shared.create(id: "bertProcessing", parent: "generateAudioForTokens")
          
          // Ensure model is initialized
          guard let bert = bert,
                let bertEncoder = bertEncoder else {
            throw KokoroTTSError.modelNotInitialized
          }

          let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)

          autoreleasepool {
            _ = attentionMask
          }

          let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
          
          BenchmarkTimer.shared.stop(id: "bertProcessing")

          autoreleasepool {
            _ = bertDur
          }

          var refS: MLXArray
          do {
            guard let voice = voice else {
              throw KokoroTTSError.modelNotInitialized
            }
            refS = voice[min(inputIds.count - 1, voice.shape[0] - 1), 0 ... 1, 0...]
          } catch {
            // Use a fallback slice from start of the voice array
            guard let voice = voice else {
              throw KokoroTTSError.modelNotInitialized
            }
            refS = voice[0, 0 ... 1, 0...]
          }

          let s = refS[0 ... 1, 128...]

          return try autoreleasepool { () -> MLXArray in
            // Time duration prediction
            BenchmarkTimer.shared.create(id: "durationPrediction", parent: "generateAudioForTokens")
            
            // Ensure all components are initialized
            guard let durationEncoder = durationEncoder,
                  let predictorLSTM = predictorLSTM,
                  let durationProj = durationProj else {
              throw KokoroTTSError.modelNotInitialized
            }

            let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)

            autoreleasepool {
              _ = dEn
              _ = textMask
            }

            let (x, _) = predictorLSTM(d)

            let duration = durationProj(x)

            autoreleasepool {
              _ = x
            }

            let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed

            autoreleasepool {
              _ = duration
            }

            let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]

            autoreleasepool {
              _ = durationSigmoid
            }
            
            BenchmarkTimer.shared.stop(id: "durationPrediction")

            // Time matrix operations (computationally expensive)
            BenchmarkTimer.shared.create(id: "matrixOperations", parent: "generateAudioForTokens")
            
            // Index and matrix generation - high memory usage
            // Build indices in chunks to reduce memory
            var allIndices: [MLXArray] = []
            let chunkSize = 50 // Process 50 items at a time

            for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
              autoreleasepool {
                let endIdx = min(startIdx + chunkSize, predDur.shape[0])
                let chunkIndices = predDur[startIdx..<endIdx]

                let indices = MLX.concatenated(
                  chunkIndices.enumerated().map { i, n in
                    let nSize: Int = n.item()
                    let arrayIndex = MLXArray([i + startIdx])
                    let repeated = MLX.repeated(arrayIndex, count: nSize)
                    return repeated
                  }
                )
                allIndices.append(indices)
              }
            }

            let indices = MLX.concatenated(allIndices)

            allIndices.removeAll()

            let indicesShape = indices.shape[0]
            let inputIdsShape = paddedInputIds.shape[1]

            // Create sparse matrix more efficiently using COO format
            // This drastically reduces memory usage compared to dense matrix
            var rowIndices: [Int] = []
            var colIndices: [Int] = []
            var values: [Float] = []

            // Reserve capacity to avoid reallocations
            let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
            rowIndices.reserveCapacity(estimatedNonZeros)
            colIndices.reserveCapacity(estimatedNonZeros)
            values.reserveCapacity(estimatedNonZeros)

            // Process in batches to reduce cache misses
            let batchSize = 256
            for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + batchSize, indicesShape)
                for i in startIdx..<endIdx {
                  let indiceValue: Int = indices[i].item()
                  if indiceValue < inputIdsShape {
                    rowIndices.append(indiceValue)
                    colIndices.append(i)
                    values.append(1.0)
                  }
                }
              }
            }

            autoreleasepool {
              _ = indices
              _ = predDur
            }

            // Create MLXArray from COO data
            let rowIndicesArray = MLXArray(rowIndices)
            let colIndicesArray = MLXArray(colIndices)
            let coo_indices = MLX.stacked([rowIndicesArray, colIndicesArray], axis: 0).transposed(1, 0)
            let coo_values = MLXArray(values)

            // Go back to the original dense matrix approach but with better memory management
            // Create sparse matrix efficiently using Swift arrays first
            var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
            // Process in batches to reduce cache misses
            let matrixBatchSize = 1000
            for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
                for i in startIdx..<endIdx {
                  let row = rowIndices[i]
                  let col = colIndices[i]
                  if row < inputIdsShape && col < indicesShape {
                    swiftPredAlnTrg[row * indicesShape + col] = 1.0
                  }
                }
              }
            }

            // Create MLXArray from the dense matrix
            let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])

            // Clear Swift array immediately
            swiftPredAlnTrg = []

            autoreleasepool {
              rowIndices = []
              colIndices = []
              values = []
            }

            let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)

            let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)

            autoreleasepool {
              _ = d
              _ = predAlnTrgBatched
            }
            
            BenchmarkTimer.shared.stop(id: "matrixOperations")

            return try autoreleasepool { () -> MLXArray in
              // Time final audio decoding
              BenchmarkTimer.shared.create(id: "audioDecoding", parent: "generateAudioForTokens")
              
              // Ensure components are initialized
              guard let prosodyPredictor = prosodyPredictor,
                    let textEncoder = textEncoder,
                    let decoder = decoder else {
                throw KokoroTTSError.modelNotInitialized
              }

              let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)

              autoreleasepool {
                _ = en
              }

              let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)

              autoreleasepool {
                _ = paddedInputIds
                _ = inputLengths
              }

              let asr = MLX.matmul(tEn, predAlnTrg)

              autoreleasepool {
                _ = tEn
                _ = predAlnTrg
              }

              let voiceS = refS[0 ... 1, 0 ... 127]

              autoreleasepool {
                _ = refS
              }

              let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]

              autoreleasepool {
                _ = asr
                _ = F0Pred
                _ = NPred
                _ = voiceS
                _ = s
              }
              
              BenchmarkTimer.shared.stop(id: "audioDecoding")

              let audioShape = audio.shape

              // Check if the audio shape is valid
              let totalSamples: Int
              if audioShape.count == 1 {
                totalSamples = audioShape[0]
              } else if audioShape.count == 2 {
                totalSamples = audioShape[1]
              } else {
                totalSamples = 0
              }

              if totalSamples <= 1 {
                // Return an error tone
                var errorAudioData = [Float](repeating: 0.0, count: 24000)

                // Create a simple repeating beep pattern to indicate error
                for i in stride(from: 0, to: 24000, by: 100) {
                  let endIdx = min(i + 100, 24000)
                  for j in i..<endIdx {
                    let t = Float(j) / Float(Constants.sampleRate)
                    let freq = (Int(t * 2) % 2 == 0) ? 500.0 : 800.0
                    errorAudioData[j] = sin(Float(2.0 * .pi * freq) * t) * 0.5
                  }
                }

                let fallbackAudio = MLXArray(errorAudioData)
                return fallbackAudio
              }

              return audio
            }
          }
        }
      }
    }
  }

  // New async stream interface - returns when generation is complete
  public func generateAudioStream(voice: TTSVoice, text: String, speed: Float = 1.0) -> AsyncThrowingStream<MLXArray, Error> {
    return AsyncThrowingStream { continuation in
      Task {
        do {
          try ensureModelInitialized()
          
          let sentences = SentenceTokenizer.splitIntoSentences(text: text)
          guard !sentences.isEmpty else {
            throw KokoroTTSError.sentenceSplitError
          }
          
          // Reset voice for new generation
//          self.voice = nil
          
          // Process each sentence sequentially
          for sentence in sentences {
            autoreleasepool {
              do {
                let audio = try self.generateAudioForSentence(voice: voice, text: sentence, speed: speed)
                
                // Yield the audio chunk
                continuation.yield(audio)
                
                // Clean up
                autoreleasepool {
                  _ = audio
                }
              } catch {
                continuation.finish(throwing: error)
                return
              }
            }

          }
          
          // All sentences processed successfully
          continuation.finish()
          
          // Reset model after completing long text to free memory
          if sentences.count > 5 {
            Task {
              try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
              self.resetModel()
            }
          }
          
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }
  
  // Legacy callback-based method - deprecated, use generateAudioStream instead
  @available(*, deprecated, message: "Use generateAudioStream instead for better async control")
  public func generateAudio(voice: TTSVoice, text: String, speed: Float = 1.0, chunkCallback: @escaping AudioChunkCallback) throws {
    try ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    // Process each sentence in sequence with better performance
    DispatchQueue.global(qos: .userInitiated).async {
      self.voice = nil

      // Use a separate autorelease pool for each sentence to release memory faster
      for (_, sentence) in sentences.enumerated() {
        autoreleasepool {
          do {
            // Generate audio for this sentence
            let audio = try self.generateAudioForSentence(voice: voice, text: sentence, speed: speed)

            // Send this chunk to the callback immediately on the main thread
            // Dispatch to main thread to avoid threading issues with UI updates
            DispatchQueue.main.async {
              chunkCallback(audio)
            }

            // Explicitly release large tensors
            autoreleasepool {
              _ = audio
            }
          } catch {
            // Handle error silently
          }
        }
        MLX.GPU.clearCache()
      }

      // Reset model after completing a long text to free memory
      if sentences.count > 5 {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.resetModel()
        }
      }
    }
  }

  private func generateAudioForSentence(voice: TTSVoice, text: String, speed: Float) throws -> MLXArray {
    // Reset timer for clean measurements
    BenchmarkTimer.shared.reset()
    
    // Start timing the entire method
    BenchmarkTimer.shared.create(id: "generateAudioForSentence")
    defer {
      BenchmarkTimer.shared.stop(id: "generateAudioForSentence")
      BenchmarkTimer.shared.logResults(id: "generateAudioForSentence")
    }
    
    try ensureModelInitialized()

    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return MLXArray.zeros([1])
    }

    return try autoreleasepool { () -> MLXArray in
      if chosenVoice != voice {
        // Time voice loading
        BenchmarkTimer.shared.create(id: "voiceLoading", parent: "generateAudioForSentence")
        autoreleasepool {
          self.voice = VoiceLoader.loadVoice(voice)
        }

        try kokoroTokenizer.setLanguage(for: voice)
        chosenVoice = voice
        BenchmarkTimer.shared.stop(id: "voiceLoading")
      }

      do {
        // Time text processing (phonemization and tokenization)
        BenchmarkTimer.shared.create(id: "textProcessing", parent: "generateAudioForSentence")
        let phonemizedResult = try kokoroTokenizer.phonemize(text)
        let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)
        BenchmarkTimer.shared.stop(id: "textProcessing")
        
        guard inputIds.count <= Constants.maxTokenCount else {
          throw KokoroTTSError.tooManyTokens
        }

        // Time audio generation
        BenchmarkTimer.shared.create(id: "audioGeneration", parent: "generateAudioForSentence")
        let result = try self.processTokensToAudio(inputIds: inputIds, speed: speed)
        result.eval()
        BenchmarkTimer.shared.stop(id: "audioGeneration")

        return result
      } catch {
        // Return a short error tone instead of crashing
        var errorAudioData = [Float](repeating: 0.0, count: 4800) // 0.2s at 24kHz

        // Simple error beep
        for i in 0..<4800 {
          let t = Float(i) / Float(Constants.sampleRate)
          let freq: Float = 880.0 // High-pitched error tone
          errorAudioData[i] = sin(Float(2.0 * .pi * freq) * t) * 0.3
        }

        return MLXArray(errorAudioData)
      }
    }
  }

  // Common processing method to convert tokens to audio - used by streaming methods
  private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> MLXArray {
    // Use the token processing method
    return try generateAudioForTokens(
      inputIds: inputIds,
      speed: speed
    )
  }

  struct Constants {
    static let maxTokenCount = 510
    static let sampleRate = 24000
  }
}
