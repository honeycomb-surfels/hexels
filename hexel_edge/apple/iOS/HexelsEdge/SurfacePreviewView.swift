import SceneKit
import SwiftUI

struct SurfacePreviewItem: Identifiable {
    let url: URL
    let title: String
    var id: String { url.absoluteString }
}

/// Inspect the actual captured geometry; it does not generate a new surface.
struct SurfacePreviewView: View {
    let item: SurfacePreviewItem
    @State private var scene: SCNScene?
    @State private var errorMessage: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let scene {
                    SceneView(scene: scene, options: [.allowsCameraControl, .autoenablesDefaultLighting],
                              preferredFramesPerSecond: 30)
                        .background(Color.black)
                        .accessibilityLabel("Captured surface. Drag to rotate and pinch to zoom.")
                } else if let errorMessage {
                    ContentUnavailableView("Preview unavailable", systemImage: "cube.transparent",
                                           description: Text(errorMessage))
                } else { ProgressView("Opening surface…") }
            }
            .navigationTitle(item.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .topBarLeading) {
                    ShareLink(item: item.url) { Label("Export", systemImage: "square.and.arrow.up") }
                }
            }
            .task { loadScene() }
        }
    }

    private func loadScene() {
        do {
            let loaded = try SCNScene(url: item.url, options: [.checkConsistency: true, .createNormalsIfAbsent: true])
            let bounds = loaded.rootNode.boundingBox
            let center = SCNVector3((bounds.min.x + bounds.max.x) / 2,
                                    (bounds.min.y + bounds.max.y) / 2,
                                    (bounds.min.z + bounds.max.z) / 2)
            let radius = max(0.05, max(bounds.max.x - bounds.min.x,
                                      max(bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z)) / 2)
            let camera = SCNNode()
            camera.camera = SCNCamera()
            camera.camera?.zNear = max(0.001, Double(radius) / 1000)
            camera.camera?.zFar = max(100, Double(radius) * 100)
            camera.position = SCNVector3(center.x + radius * 1.8,
                                         center.y + radius * 1.0, center.z + radius * 2.8)
            camera.look(at: center)
            loaded.rootNode.addChildNode(camera)
            scene = loaded
        } catch { errorMessage = error.localizedDescription }
    }
}
