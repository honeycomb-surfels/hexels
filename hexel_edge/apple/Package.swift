// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HexelsEdge",
    platforms: [.macOS(.v14), .iOS(.v18)],
    products: [
        .library(name: "HexelsMetal", targets: ["HexelsMetal"]),
        .executable(name: "hexels-bench", targets: ["HexelsBench"]),
        .executable(name: "hexels-tests", targets: ["HexelsMetalTests"]),
    ],
    targets: [
        .target(name: "HexelsMetal", resources: [.copy("Hexels.metal")]),
        .executableTarget(name: "HexelsBench", dependencies: ["HexelsMetal"]),
        .executableTarget(name: "HexelsMetalTests", dependencies: ["HexelsMetal"], path: "Tests/HexelsMetalTests"),
    ],
    swiftLanguageModes: [.v5]
)
