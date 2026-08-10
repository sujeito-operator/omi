import XCTest

@testable import Omi_Computer

private actor SessionStepRecorder {
  private(set) var steps: [String] = []

  func record(_ step: String) {
    steps.append(step)
  }
}

final class AgentVMSessionStartupTests: XCTestCase {
  /// Removing screen egress must not remove the sanitized non-screen context
  /// path: a ready VM still has to start `AgentSyncService`, and only after the
  /// VM's screen activity has been purged.
  func testReadySessionPurgesScreenActivityThenStartsNonScreenSync() async {
    let recorder = SessionStepRecorder()
    let service = AgentVMService(
      sessionHooks: AgentVMService.SessionHooks(
        purgeScreenActivity: { _, _ in
          await recorder.record("purge")
          return true
        },
        sendFirebaseToken: { _, _ in await recorder.record("token") },
        startNonScreenSync: { _, _ in await recorder.record("sync") }))

    await service.startAgentSession(vmIP: "10.0.0.1", authToken: "token")

    let steps = await recorder.steps
    XCTAssertEqual(steps, ["purge", "token", "sync"])
  }

  func testFailedScreenPurgeBlocksNonScreenSync() async {
    let recorder = SessionStepRecorder()
    let service = AgentVMService(
      sessionHooks: AgentVMService.SessionHooks(
        purgeScreenActivity: { _, _ in
          await recorder.record("purge")
          return false
        },
        sendFirebaseToken: { _, _ in await recorder.record("token") },
        startNonScreenSync: { _, _ in await recorder.record("sync") }))

    await service.startAgentSession(vmIP: "10.0.0.1", authToken: "token")

    let steps = await recorder.steps
    XCTAssertEqual(steps, ["purge"])
  }
}
