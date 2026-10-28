# Relocated source verification

Verified on 2026-10-28 with Xcode 27.1 / Swift 6.4. Builds used isolated temporary
directories; no physical-device installation or camera interaction occurred.

| Check | Result |
|---|---|
| Relocated Release Swift package, engine and CLI | Passed |
| Independent native CPU/Metal checks | 30 passed, 0 failed, 0 skipped; real Metal required |
| Actual CoreVideo/pixel/PNG/bundle/surface/cancel fixtures | 29 passed using synthetic private temporary data |
| Pure device-policy checks | Passed |
| Python preparation/quality/report suite | 35 passed; no dataset downloads or training |
| Final unsigned generic iOS Release build | Passed with iPhoneOS 27.1 SDK |
| Built app metadata | Display name Hexel Edge; camera usage description; File Sharing enabled; one scene; no embedded provisioning profile |
| Packaged Metal resource | Byte-identical to source |
| Canonical icon | Byte-identical to assets/app-icon.png, opaque 1024×1024 RGB |

The engine/CLI source and shaders are byte-identical to the local development
source. Public app changes are display branding and the canonical icon; a test
uses the system temporary directory. Public guides use relative project paths.

The package excludes datasets, outputs, private captures/reports, checkpoints,
measured archives, build products, personal signing configuration, Xcode user
state and AppleDouble metadata. The signing example contains placeholders only.
The shared bundle identifier is the generic `io.hexels.edgeprototype`; users
should choose their own identifier/team where required. No account email,
personal Team ID, UDID, container path or workstation home path was retained.

The icon SHA256 is
`e13bd674423c352bc5ba808a873cafb42f3a0eb331ebd36ef60d208a1008da0d`.
See [engine commands](README.md), [iOS build/setup](iOS/README.md),
[capture fixture commands](iOS/Tests/README.md) and
[Python tool checks](../tools/README.md) for reproduction.

These checks validate relocation and the declared synthetic/numerical contracts.
An unsigned SDK build does not establish installation, camera behavior or
performance on a new physical device.
