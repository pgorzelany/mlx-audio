//
//  Kokoro-tts-lib
//
import Foundation
import OSLog

class BenchmarkTimer {
  class Timing {
    let id: String
    private let logger = Logger(subsystem: "com.babaru", category: "kokoro-tts")

    private var start: DispatchTime
    private var finish: DispatchTime?
    private var childTasks: [Timing] = []
    private let parent: Timing?
    private var delta: UInt64 = 0

    init(id: String, parent: Timing?) {
      start = DispatchTime.now()
      self.id = id
      self.parent = parent
      if let parent { parent.childTasks.append(self) }
    }

    func startTimer() {
      start = DispatchTime.now()
    }

    func stop() {
      finish = DispatchTime.now()
      delta += finish!.uptimeNanoseconds - start.uptimeNanoseconds
    }

    func logTiming(spaces: Int = 0) {
      guard let _ = finish else { return }

      let spaceString = String(repeating: " ", count: spaces)
      let message = "\(spaceString)\(id): \(deltaInSec) sec"
      
      // Use OSLog with appropriate log level
      logger.info("\(message, privacy: .public)")
      
      for childTask in childTasks {
        childTask.logTiming(spaces: spaces + 2)
      }
    }

    var deltaTime: Double { Double(delta) / 1_000_000_000 }
    var deltaInSec: String { String(format: "%.4f", Double(delta) / 1_000_000_000) }
  }

  static let shared = BenchmarkTimer()
  private let logger = Logger(subsystem: "com.babaru", category: "kokoro-tts")

  private init() {}

  private var timers: [String: Timing] = [:]

  @discardableResult
  func create(id: String, parent parentId: String? = nil) -> Timing? {
    guard timers[id] == nil else { return nil }

    var parentTiming: Timing?
    if let parentId {
      parentTiming = timers[parentId]
      guard parentTiming != nil else { return nil }
    }

    timers[id] = Timing(id: id, parent: parentTiming)
    return timers[id]
  }

  func stop(id: String) {
    guard let timing = timers[id] else { return }
    timing.stop()
  }

  func logResults(id: String) {
    guard let timing = timers[id] else { return }
    logger.info("📊 Performance Timing Results:")
    timing.logTiming()
  }
  
  // Deprecated method for backward compatibility
  @available(*, deprecated, message: "Use logResults instead")
  func printLog(id: String) {
    logResults(id: id)
  }

  func reset() {
    timers = [:]
  }
}
