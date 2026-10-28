import Foundation
import HexelsMetal
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Darwin

struct Options {
    let command: String
    var values: [String:String] = [:]
    init() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let first = args.first, ["train","render","inspect"].contains(first) else {
            throw HexelsError.invalid("Usage: hexels-bench train|render|inspect --bundle PATH [--output PATH] [--steps 1000] [--mode full|appearance] [--recipe standard|detail] [--resume checkpoint.hextrain] [--memory-mib 512] [--max-intersections 2000000] [--checkpoint-every 250]")
        }
        command = first
        let allowed: Set<String> = ["bundle","output","steps","mode","recipe","resume","memory-mib","max-intersections","checkpoint-every"]
        var index = 1
        while index < args.count {
            let key = String(args[index].dropFirst(2))
            guard args[index].hasPrefix("--"), allowed.contains(key), index+1 < args.count, values[key] == nil else {
                throw HexelsError.invalid("Unknown, duplicate, or incomplete option: \(args[index])")
            }
            values[key] = args[index+1]; index += 2
        }
        guard values["bundle"] != nil else { throw HexelsError.invalid("--bundle is required") }
    }
    func positive(_ key:String, default fallback:Int) throws -> Int {
        guard let value = values[key] else { return fallback }
        guard let number = Int(value), number > 0 else { throw HexelsError.invalid("--\(key) must be a positive integer") }
        return number
    }
}

func record(_ object: [String:Any], to file: FileHandle? = nil) throws {
    let data = try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys]) + Data([10])
    FileHandle.standardOutput.write(data)
    try file?.write(contentsOf:data)
}

func footprint() -> (current:UInt64, peak:UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to:&info) { pointer in
        pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
            task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
        }
    }
    return status == KERN_SUCCESS ? (UInt64(info.phys_footprint),UInt64(info.ledger_phys_footprint_peak)) : (0,0)
}

func writePNG(_ pixels:[SIMD4<Float>], width:Int, height:Int, to url:URL) throws {
    var bytes = [UInt8](); bytes.reserveCapacity(width*height*4)
    for p in pixels {
        for value in [p.x,p.y,p.z] { bytes.append(UInt8(min(255,max(0,(value*255).rounded())))) }
        bytes.append(255)
    }
    let data = Data(bytes)
    guard let provider = CGDataProvider(data:data as CFData), let space = CGColorSpace(name:CGColorSpace.sRGB),
          let image = CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:width*4,
            space:space,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.last.rawValue),provider:provider,
            decode:nil,shouldInterpolate:false,intent:.defaultIntent),
          let destination = CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil) else {
        throw HexelsError.invalid("Cannot encode output PNG")
    }
    CGImageDestinationAddImage(destination,image,nil)
    guard CGImageDestinationFinalize(destination) else { throw HexelsError.invalid("Cannot save output PNG") }
}

func evaluate(_ trainer:MetalHexelTrainer, bundle:SceneBundle, output:URL?, prefix:String) throws -> [String:Any] {
    let test = bundle.frames.filter { $0.split == "test" }
    var squared: Double = 0, samples = 0
    var perView:[Double] = []
    for (index,frame) in test.enumerated() {
        let prediction = try trainer.render(camera:frame.camera,background:bundle.background)
        let target = try bundle.image(for:frame)
        var viewSquared:Double = 0
        for i in prediction.indices {
            let d = prediction[i]-target[i]
            viewSquared += Double(d.x*d.x+d.y*d.y+d.z*d.z); samples += 3
        }
        squared += viewSquared
        perView.append(-10*log10(max(viewSquared/Double(target.count*3),1e-20)))
        if let output {
            try writePNG(prediction,width:frame.width,height:frame.height,
                         to:output.appendingPathComponent("\(prefix)_\(index).png"))
        }
    }
    guard samples > 0 else { return ["views":0] }
    let mse = squared/Double(samples)
    return ["views":test.count,"rgb_mse":mse,"psnr_db":-10*log10(max(mse,1e-20)),
        "psnr_aggregation":"pooled RGB MSE over all held-out pixels",
        "mean_view_psnr_db":perView.reduce(0,+)/Double(perView.count),"per_view_psnr_db":perView]
}

func main() throws {
    let options = try Options()
    let bundle = try SceneBundle(url:URL(fileURLWithPath:options.values["bundle"]!,isDirectory:true))
    if options.command == "inspect" {
        try record(["schema":1,"primitives":bundle.initialParameters.count/20,"train_views":bundle.frames.filter{$0.split=="train"}.count,
            "test_views":bundle.frames.filter{$0.split=="test"}.count,"sampling_steps":bundle.viewOrder.count])
        return
    }
    let output = URL(fileURLWithPath:options.values["output"] ?? "edge-run",isDirectory:true)
    guard !FileManager.default.fileExists(atPath:output.path) else {
        throw HexelsError.invalid("Use a fresh --output directory; refusing to overwrite \(output.path)")
    }
    let steps = try options.positive("steps",default:1000)
    guard steps <= 1_000_000 else { throw HexelsError.invalid("Step budget exceeds one million") }
    let mode = options.values["mode"] ?? "full"
    guard mode == "full" || mode == "appearance" else { throw HexelsError.invalid("--mode must be full or appearance") }
    let memoryMiB = try options.positive("memory-mib",default:512)
    guard memoryMiB <= 16384 else { throw HexelsError.invalid("--memory-mib exceeds supported admission limit") }
    let recipe = options.values["recipe"] ?? bundle.recipeHint
    guard recipe == "standard" || recipe == "detail" else { throw HexelsError.invalid("--recipe must be standard or detail") }
    var rates = recipe == "detail" ? TrainingConfiguration.detailRates : TrainingConfiguration.defaultRates
    if mode == "appearance" { for i in 0..<9 { rates[i] = 0 }; rates[19] = 0 }
    let config = TrainingConfiguration(learningRates:rates,
        maxIntersections:try options.positive("max-intersections",default:2_000_000),maxWorkingSetBytes:memoryMiB*1024*1024)
    let trainer: MetalHexelTrainer
    if let resume = options.values["resume"] {
        trainer = try MetalHexelTrainer.loadCheckpoint(from:URL(fileURLWithPath:resume),configuration:config)
        guard trainer.checkpointContext == bundle.fingerprint else { throw HexelsError.invalid("Checkpoint belongs to a different or unverified bundle") }
        guard trainer.learningRates == rates else { throw HexelsError.invalid("Requested mode/rates differ from the checkpoint; resume with the original mode") }
    }
    else {
        trainer = try MetalHexelTrainer(parameters:bundle.initialParameters,configuration:config)
        trainer.checkpointContext = bundle.fingerprint
    }
    guard trainer.parameters.count == bundle.initialParameters.count else { throw HexelsError.invalid("Checkpoint population differs from bundle") }
    let checkpointEvery = try options.positive("checkpoint-every",default:250)
    try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    let logURL = output.appendingPathComponent("events.jsonl")
    FileManager.default.createFile(atPath:logURL.path,contents:nil)
    let log = try FileHandle(forWritingTo:logURL); defer { try? log.close() }
    try record(["event":"start","device":trainer.deviceName,"system":ProcessInfo.processInfo.operatingSystemVersionString,
        "physical_memory_bytes":ProcessInfo.processInfo.physicalMemory,"mode":mode,"recipe":recipe,"primitives":trainer.parameters.count/20,
        "start_step":trainer.stepCount,"requested_steps":steps,"memory_admission_bytes":config.maxWorkingSetBytes,
        "bundle_sha256":bundle.fingerprint],to:log)
    let initial = try evaluate(trainer,bundle:bundle,output:output,prefix:"initial")
    if options.command == "render" {
        try record(["event":"evaluation","metrics":initial],to:log); return
    }
    var times:[Double] = [], peakBytes:UInt64 = 0, peakGPU = 0, peakPairs = 0
    let started = Date(), initialStep = trainer.stepCount
    var initialLoss:Float = 0, finalLoss:Float = 0
    // At most two decoded training targets. RGB targets use encoded sRGB values,
    // matching benchmark tensors rather than silently linearizing their transfer.
    var cache:[Int:[SIMD4<Float>]] = [:], cacheOrder:[Int] = []
    for iteration in 0..<steps {
        let t = Date()
        let frameID = Int(bundle.viewOrder[trainer.stepCount % bundle.viewOrder.count])
        let frame = bundle.frames[frameID]
        let target:[SIMD4<Float>]
        if let stored = cache[frameID] { target = stored }
        else {
            target = try bundle.image(for:frame)
            if cacheOrder.count == 2 { cache.removeValue(forKey:cacheOrder.removeFirst()) }
            cache[frameID] = target; cacheOrder.append(frameID)
        }
        let loss = try trainer.step(camera:frame.camera,target:target,background:bundle.background)
        if iteration == 0 { initialLoss = loss }; finalLoss = loss
        times.append(Date().timeIntervalSince(t))
        let memory = footprint(); peakBytes=max(peakBytes,max(memory.current,memory.peak))
        peakGPU=max(peakGPU,trainer.peakAllocatedGPUBytes); peakPairs=max(peakPairs,trainer.lastIntersectionCount)
        if trainer.stepCount % checkpointEvery == 0 { try trainer.saveCheckpoint(to:output.appendingPathComponent("checkpoint.hextrain")) }
        if iteration % max(1,min(50,steps/10)) == 0 || iteration == steps-1 {
            try record(["event":"step","step":trainer.stepCount,"loss":loss,"wall_seconds":Date().timeIntervalSince(started),
                "step_seconds":times.last!,"physical_footprint_bytes":memory.current,
                "forward_gpu_seconds":trainer.lastForwardGPUSeconds,"backward_gpu_seconds":trainer.lastBackwardGPUSeconds,
                "metal_allocated_bytes":trainer.allocatedGPUBytes,"tile_intersections":trainer.lastIntersectionCount,
                "thermal_state":ProcessInfo.processInfo.thermalState.rawValue],to:log)
        }
    }
    let trainingSeconds = Date().timeIntervalSince(started)
    try trainer.saveCheckpoint(to:output.appendingPathComponent("checkpoint.hextrain"))
    let final = try evaluate(trainer,bundle:bundle,output:output,prefix:"final")
    let sorted = times.sorted(), warm = Array(times.dropFirst(min(10,times.count/4))).sorted()
    let summary:[String:Any] = ["schema":1,"device":trainer.deviceName,"start_step":initialStep,"end_step":trainer.stepCount,
        "training_seconds":trainingSeconds,"mean_step_seconds":times.reduce(0,+)/Double(times.count),
        "median_step_seconds":sorted[sorted.count/2],"warm_median_step_seconds":warm[warm.count/2],
        "p95_step_seconds":sorted[min(sorted.count-1,Int(Double(sorted.count)*0.95))],
        "peak_process_physical_footprint_bytes":peakBytes,"sampled_peak_metal_allocated_bytes":peakGPU,
        "peak_tile_intersections":peakPairs,"first_sampled_training_loss":initialLoss,"last_sampled_training_loss":finalLoss,
        "initial_test":initial,"final_test":final,"mode":mode,"recipe":recipe,"measured_platform":"macOS",
        "caveat":"Mac measurement; iPad/iPhone performance and deployment remain unvalidated. CPU tile binning/sorting included."]
    try JSONSerialization.data(withJSONObject:summary,options:[.prettyPrinted,.sortedKeys]).write(to:output.appendingPathComponent("summary.json"),options:.atomic)
    try record(["event":"complete","summary":summary],to:log)
}

do { try main() }
catch {
    let message = "hexels-bench: \(error.localizedDescription)\n"
    FileHandle.standardError.write(Data(message.utf8)); exit(1)
}
