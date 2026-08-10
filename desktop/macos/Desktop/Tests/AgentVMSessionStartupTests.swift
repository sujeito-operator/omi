import XCTest

@testable import Omi_Computer

private actor SessionStepRecorder {
  private(set) var steps: [String] = []

  func record(_ step: String) {
    steps.append(step)
  }
}

/// Closing screen egress must not disconnect the sanitized non-screen context
/// path, and must not let anything reach the VM before its screen activity is
/// purged. These drive the real session preparation with injected hooks.
final class AgentVMSessionStartupTests: XCTestCase {
  private var ownerFixture: RuntimeOwnerAuthorityTestFixture?
  private let ownerID = "agent-vm-session-owner"

  override func setUp() async throws {
    try await super.setUp()
    ownerFixture = RuntimeOwnerAuthorityTestFixture()
    await ownerFixture?.establish(authOwnerID: ownerID)
  }

  override func tearDown() async throws {
    await ownerFixture?.restore()
    ownerFixture = nil
    try await super.tearDown()
  }

  private func makeService(
    recorder: SessionStepRecorder,
    purgeSucceeds: Bool
  ) -> AgentVMService {
    AgentVMService(
      sessionHooks: AgentVMService.SessionHooks(
        purgeScreenActivity: { _, _ in
          await recorder.record("purge")
          return purgeSucceeds
        },
        startNonScreenSync: { _, _ in await recorder.record("sync") },
        sendFirebaseToken: { _, _, _, _ in await recorder.record("token") }))
  }

  private func readyStatus() -> APIClient.AgentStatusResponse {
    APIClient.AgentStatusResponse(
      vmName: "vm-test",
      zone: "",
      ip: "10.0.0.1",
      status: "ready",
      authToken: "vm-token",
      createdAt: "",
      lastQueryAt: nil)
  }

  func testReadyVMPurgesScreenActivityThenStartsNonScreenSync() async {
    let recorder = SessionStepRecorder()
    let service = makeService(recorder: recorder, purgeSucceeds: true)

    await service.prepareReadyVM(readyStatus(), ip: "10.0.0.1", ownerID: ownerID, generation: 0)

    let steps = await recorder.steps
    XCTAssertEqual(steps, ["purge", "sync", "token"])
  }

  func testFailedScreenPurgeBlocksSyncAndBackendToken() async {
    let recorder = SessionStepRecorder()
    let service = makeService(recorder: recorder, purgeSucceeds: false)

    await service.prepareReadyVM(readyStatus(), ip: "10.0.0.1", ownerID: ownerID, generation: 0)

    let steps = await recorder.steps
    XCTAssertEqual(steps, ["purge"])
  }

  /// Full-database upload is retired because the local database can hold
  /// screen/OCR rows; the sync-failure recovery hook must not reopen it.
  func testDatabaseReuploadStaysDisabled() async {
    let recorder = SessionStepRecorder()
    let service = makeService(recorder: recorder, purgeSucceeds: true)

    let reuploaded = await service.reuploadDatabase(vmIP: "10.0.0.1", authToken: "vm-token")

    XCTAssertFalse(reuploaded)
  }
}
