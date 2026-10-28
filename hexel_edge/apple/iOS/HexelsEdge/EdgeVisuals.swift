import SwiftUI
import UIKit

extension View {
    func edgeCard() -> some View {
        self.padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 24))
            .overlay(RoundedRectangle(cornerRadius: 24).strokeBorder(.primary.opacity(0.05)))
    }
}

struct HexelsMark: View {
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            ZStack {
                HexagonShape().fill(LinearGradient(colors: [.indigo, .blue], startPoint: .topLeading, endPoint: .bottomTrailing))
                Path { path in
                    path.move(to: CGPoint(x: width * 0.24, y: height * 0.37))
                    path.addLine(to: CGPoint(x: width * 0.5, y: height * 0.52))
                    path.addLine(to: CGPoint(x: width * 0.76, y: height * 0.37))
                    path.move(to: CGPoint(x: width * 0.5, y: height * 0.52))
                    path.addLine(to: CGPoint(x: width * 0.5, y: height * 0.8))
                }
                .stroke(.white.opacity(0.95), style: StrokeStyle(lineWidth: max(1.5, width * 0.065), lineCap: .round, lineJoin: .round))
            }
        }
        .accessibilityHidden(true)
    }
}

struct HexagonShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        for index in 0..<6 {
            let angle = Double(index) * .pi / 3 - .pi / 2
            let point = CGPoint(x: rect.midX + cos(angle) * rect.width / 2,
                                y: rect.midY + sin(angle) * rect.height / 2)
            if index == 0 { path.move(to: point) } else { path.addLine(to: point) }
        }
        path.closeSubpath()
        return path
    }
}

struct HexelsSceneIllustration: View {
    var body: some View {
        ZStack {
            Ellipse().fill(Color.indigo.opacity(0.1)).frame(width: 148, height: 28).blur(radius: 12).offset(y: 62)
            ForEach(0..<7) { index in
                let angle = Double(index) * .pi / 3
                let x = index == 6 ? 0 : cos(angle) * 54
                let y = index == 6 ? 0 : sin(angle) * 44
                HexagonShape()
                    .fill(Color.indigo.opacity(index == 6 ? 0.16 : 0.06))
                    .overlay(HexagonShape().stroke(Color.indigo.opacity(index == 6 ? 0.45 : 0.22), lineWidth: 1.2))
                    .frame(width: 54, height: 64).offset(x: x, y: y)
            }
            HexelsMark().frame(width: 42, height: 48)
        }
        .rotation3DEffect(.degrees(15), axis: (x: 1, y: -0.25, z: 0))
        .accessibilityHidden(true)
    }
}

struct EdgeMetric: View {
    let title: String
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(value).font(.title3.weight(.semibold).monospacedDigit())
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), \(value)")
    }
}

struct EdgeInfoRow: View {
    let title: String
    let value: String
    let symbol: String
    var color: Color = .secondary
    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: symbol).foregroundStyle(color).frame(width: 18)
            Text(title).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value).fontWeight(.medium).monospacedDigit()
        }
        .font(.caption)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), \(value)")
    }
}

/// Two-dimensional inspection only; camera changes are explicit in the parent UI.
struct ZoomableSceneImage: View {
    let image: UIImage
    @Binding var isZoomed: Bool
    @State private var scale: CGFloat = 1
    @State private var lastScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    private var zoomLabel: String { String(format: "%.1f", Double(scale)) }

    var body: some View {
        GeometryReader { geometry in
            Image(uiImage: image).resizable().interpolation(.high).scaledToFit()
                .frame(width: geometry.size.width, height: geometry.size.height)
                .scaleEffect(scale).offset(offset)
                .gesture(MagnificationGesture().onChanged { factor in
                    scale = min(4, max(1, lastScale * factor))
                    constrainOffset(in: geometry.size)
                }.onEnded { _ in
                    lastScale = scale
                    if scale == 1 { offset = .zero; lastOffset = .zero }
                    lastOffset = offset
                })
                .simultaneousGesture(DragGesture(minimumDistance: scale > 1 ? 0 : 10_000).onChanged { value in
                    offset = CGSize(width: lastOffset.width + value.translation.width,
                                    height: lastOffset.height + value.translation.height)
                    constrainOffset(in: geometry.size)
                }.onEnded { _ in lastOffset = offset })
                .onTapGesture(count: 2) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        scale = scale > 1 ? 1 : 2
                        lastScale = scale; offset = .zero; lastOffset = .zero
                    }
                }
                .accessibilityLabel("Scene image")
                .accessibilityHint("Pinch or double tap to zoom. Use Next view to change cameras.")
                .accessibilityValue("\(zoomLabel) times zoom")
                .accessibilityAction(named: "Zoom in") { scale = min(4, scale * 1.5); lastScale = scale }
                .accessibilityAction(named: "Reset zoom") {
                    scale = 1; lastScale = 1; offset = .zero; lastOffset = .zero
                }
                .onChange(of: geometry.size) { _, _ in
                    constrainOffset(in: geometry.size); lastOffset = offset
                }
                .onChange(of: scale) { _, value in isZoomed = value > 1.01 }
                .onDisappear { isZoomed = false }
        }
        .aspectRatio(image.size.width / max(1, image.size.height), contentMode: .fit)
        .frame(maxHeight: 540)
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(alignment: .bottomTrailing) {
            if scale > 1.01 {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        scale = 1; lastScale = 1; offset = .zero; lastOffset = .zero
                    }
                } label: {
                    Label("Reset \(zoomLabel)×", systemImage: "arrow.down.right.and.arrow.up.left")
                        .font(.caption.monospacedDigit()).foregroundStyle(.white)
                        .padding(.horizontal, 10).padding(.vertical, 7)
                        .background(.black.opacity(0.6), in: Capsule())
                }
                .buttonStyle(.plain).padding(10).accessibilityLabel("Reset image zoom")
            }
        }
    }

    private func constrainOffset(in size: CGSize) {
        let horizontal = size.width * (scale - 1) / 2
        let vertical = size.height * (scale - 1) / 2
        offset.width = min(horizontal, max(-horizontal, offset.width))
        offset.height = min(vertical, max(-vertical, offset.height))
    }
}
