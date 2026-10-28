#include <metal_stdlib>

using namespace metal;

// V1, isotropic softness, fixed population. Each primitive is 20 contiguous
// floats: center[3], logScale[2], quaternionWXYZ[4], RGB[3], gradU[3],
// gradV[3], opacityLogit, logSoftness. Host code supplies conservative tile
// lists sorted by center depth and must rebuild them after geometry updates.
constant uint kParameterStride = 20;
constant float kAlphaCutoff = 1.0f / 255.0f;
constant float kAlphaMaximum = 0.99f;
constant float kTransmittanceCutoff = 0.0001f;
constant float kMinimumSoftness = 0.0001f;
constant float kHexSlant = 0.8660254f;

struct SceneUniforms {
    float4 cameraOrigin;
    float4 cameraRight;
    float4 cameraDown;
    float4 cameraForward;
    float4 intrinsics;       // fx, fy, cx, cy, in stored-image pixel coordinates.
    float4 backgroundNear;  // RGB, near camera-z depth.
    uint4 dimensions;       // width, height, tilesX, tileSize.
};

struct AdamUniforms {
    uint parameterCount;
    uint step;
    float beta1;
    float beta2;
    float epsilon;
    float beta1Power;
    float beta2Power;
    float padding;
};

struct HexelSample {
    float3 relativeCenter;
    float3 intersection;
    float3 axisU;
    float3 axisV;
    float3 normal;
    float4 unitQuaternion;
    float inverseQuaternionNorm;
    float2 inverseScale;
    float u;
    float v;
    float depth;
    float denominator;
    float distance;
    float2 distanceGradient;
    float opacity;
    float rawSoftness;
    float softness;
    float boundary;
    float rawAlpha;
    float alpha;
    float3 color;
};

inline float stableSigmoid(float value) {
    // Avoid overflowing exp for the opacity/soft boundary extremes.
    if (value >= 0.0f) {
        return 1.0f / (1.0f + exp(-value));
    }
    float exponential = exp(value);
    return exponential / (1.0f + exponential);
}

inline float3 cameraRay(uint2 pixel, constant SceneUniforms &scene) {
    // Integer image coordinates denote pixel centers. d is deliberately not
    // normalized: its ray parameter is camera-z depth, not Euclidean range.
    float x = (float(pixel.x) - scene.intrinsics.z) / scene.intrinsics.x;
    float y = (float(pixel.y) - scene.intrinsics.w) / scene.intrinsics.y;
    return scene.cameraRight.xyz * x + scene.cameraDown.xyz * y
        + scene.cameraForward.xyz;
}

inline void hexDistance(float u, float v,
                        thread float &distance, thread float2 &gradient) {
    float first = kHexSlant * u + 0.5f * v;
    float second = kHexSlant * u - 0.5f * v;
    // Match the CUDA edge tie convention: second slant wins a slant tie;
    // either slant wins a tie against the horizontal edge.
    bool secondWins = abs(second) >= abs(first);
    float slant = secondWins ? second : first;
    if (abs(v) > abs(slant)) {
        distance = abs(v) - 1.0f;
        gradient = float2(0.0f, v >= 0.0f ? 1.0f : -1.0f);
    } else {
        float sign = slant >= 0.0f ? 1.0f : -1.0f;
        distance = abs(slant) - 1.0f;
        gradient = float2(kHexSlant * sign,
                          (secondWins ? -0.5f : 0.5f) * sign);
    }
}

inline bool evaluateHexel(device const float *parameters, uint primitiveID,
                          float3 ray, constant SceneUniforms &scene,
                          thread HexelSample &sample) {
    device const float *row = parameters + primitiveID * kParameterStride;
    sample.relativeCenter = float3(row[0], row[1], row[2])
        - scene.cameraOrigin.xyz;
    float4 rawQuaternion = float4(row[5], row[6], row[7], row[8]);
    float squaredNorm = dot(rawQuaternion, rawQuaternion);
    // A zero quaternion is invalid input. The host validates parameters;
    // this guard also prevents undefined arithmetic in a damaged scene.
    if (!(squaredNorm > 0.0f) || !isfinite(squaredNorm)) {
        return false;
    }
    sample.inverseQuaternionNorm = rsqrt(squaredNorm);
    sample.unitQuaternion = rawQuaternion * sample.inverseQuaternionNorm;
    float w = sample.unitQuaternion.x;
    float x = sample.unitQuaternion.y;
    float y = sample.unitQuaternion.z;
    float z = sample.unitQuaternion.w;
    sample.axisU = float3(1.0f - 2.0f * (y * y + z * z),
                          2.0f * (x * y + w * z),
                          2.0f * (x * z - w * y));
    sample.axisV = float3(2.0f * (x * y - w * z),
                          1.0f - 2.0f * (x * x + z * z),
                          2.0f * (y * z + w * x));
    sample.normal = float3(2.0f * (x * z + w * y),
                           2.0f * (y * z - w * x),
                           1.0f - 2.0f * (x * x + y * y));
    sample.denominator = dot(sample.normal, ray);
    if (sample.denominator == 0.0f) {
        return false;
    }
    sample.depth = dot(sample.normal, sample.relativeCenter)
        / sample.denominator;
    if (!isfinite(sample.depth) || sample.depth < scene.backgroundNear.w) {
        return false;
    }
    sample.intersection = sample.depth * ray - sample.relativeCenter;
    sample.inverseScale = exp(-float2(row[3], row[4]));
    sample.u = dot(sample.intersection, sample.axisU) * sample.inverseScale.x;
    sample.v = dot(sample.intersection, sample.axisV) * sample.inverseScale.y;
    if (!isfinite(sample.u) || !isfinite(sample.v)) {
        return false;
    }
    hexDistance(sample.u, sample.v, sample.distance, sample.distanceGradient);
    sample.opacity = stableSigmoid(row[18]);
    sample.rawSoftness = exp(row[19]);
    sample.softness = max(sample.rawSoftness, kMinimumSoftness);
    sample.boundary = stableSigmoid(-sample.softness * sample.distance);
    sample.rawAlpha = sample.opacity * sample.boundary;
    sample.alpha = min(sample.rawAlpha, kAlphaMaximum);
    if (sample.alpha < kAlphaCutoff) {
        return false;
    }
    // V1 CUDA rendering leaves the spatial color field unclamped. Clamping a
    // displayed image is a separate operation and must not change this loss.
    sample.color = float3(row[9], row[10], row[11])
        + sample.u * float3(row[12], row[13], row[14])
        + sample.v * float3(row[15], row[16], row[17]);
    return true;
}

inline uint tileForPixel(uint2 pixel, constant SceneUniforms &scene) {
    return (pixel.y / scene.dimensions.w) * scene.dimensions.z
        + pixel.x / scene.dimensions.w;
}

kernel void renderHexels(device const float *parameters [[buffer(0)]],
                         device const uint *tilePrimitiveIDs [[buffer(1)]],
                         device const uint *tileOffsets [[buffer(2)]],
                         constant SceneUniforms &scene [[buffer(3)]],
                         device float4 *rendered [[buffer(4)]],
                         uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= scene.dimensions.x || pixel.y >= scene.dimensions.y) {
        return;
    }
    uint tile = tileForPixel(pixel, scene);
    float3 ray = cameraRay(pixel, scene);
    float3 color = float3(0.0f);
    float transmittance = 1.0f;
    for (uint entry = tileOffsets[tile]; entry < tileOffsets[tile + 1]; ++entry) {
        HexelSample sample;
        if (!evaluateHexel(parameters, tilePrimitiveIDs[entry], ray, scene, sample)) {
            continue;
        }
        float remaining = transmittance * (1.0f - sample.alpha);
        // Match CUDA: the threshold-crossing primitive is not accumulated.
        if (remaining < kTransmittanceCutoff) {
            break;
        }
        color += transmittance * sample.alpha * sample.color;
        transmittance = remaining;
    }
    color += transmittance * scene.backgroundNear.xyz;
    rendered[pixel.y * scene.dimensions.x + pixel.x]
        = float4(color, 1.0f - transmittance);
}

inline void storeGradient3(thread float *gradient, uint offset, float3 value) {
    gradient[offset] = value.x;
    gradient[offset + 1] = value.y;
    gradient[offset + 2] = value.z;
}

inline float4 quaternionVJP(thread const HexelSample &sample,
                            float3 gradientU, float3 gradientV,
                            float3 gradientNormal) {
    float w = sample.unitQuaternion.x;
    float x = sample.unitQuaternion.y;
    float y = sample.unitQuaternion.z;
    float z = sample.unitQuaternion.w;
    float4 gradient;
    gradient.x = 2.0f * (z * (gradientU.y - gradientV.x)
        + y * (gradientNormal.x - gradientU.z)
        + x * (gradientV.z - gradientNormal.y));
    gradient.y = 2.0f * (y * (gradientU.y + gradientV.x)
        + z * (gradientU.z + gradientNormal.x)
        - 2.0f * x * (gradientV.y + gradientNormal.z)
        + w * (gradientV.z - gradientNormal.y));
    gradient.z = 2.0f * (x * (gradientU.y + gradientV.x)
        - 2.0f * y * (gradientU.x + gradientNormal.z)
        + z * (gradientV.z + gradientNormal.y)
        + w * (gradientNormal.x - gradientU.z));
    gradient.w = 2.0f * (-2.0f * z * (gradientU.x + gradientV.y)
        + w * (gradientU.y - gradientV.x)
        + x * (gradientU.z + gradientNormal.x)
        + y * (gradientV.z + gradientNormal.y));
    // Chain through q = rawQuaternion / ||rawQuaternion||.
    return sample.inverseQuaternionNorm
        * (gradient - sample.unitQuaternion * dot(sample.unitQuaternion, gradient));
}

kernel void backwardHexels(device const float *parameters [[buffer(0)]],
                           device const uint *tilePrimitiveIDs [[buffer(1)]],
                           device const uint *tileOffsets [[buffer(2)]],
                           constant SceneUniforms &scene [[buffer(3)]],
                           device const float4 *rendered [[buffer(4)]],
                           device const float4 *target [[buffer(5)]],
                           device atomic_float *gradient [[buffer(6)]],
                           uint2 pixel [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    // The host launches tile-aligned 16x16 groups. All lanes of a SIMD group
    // therefore replay the same primitive list, including nonuniform edge
    // groups. Invalid pixels still participate in every SIMD reduction.
    bool active = pixel.x < scene.dimensions.x && pixel.y < scene.dimensions.y;
    uint2 safePixel = min(pixel, scene.dimensions.xy - uint2(1));
    uint pixelID = safePixel.y * scene.dimensions.x + safePixel.x;
    float3 finalColor = rendered[pixelID].xyz;
    float normalization = 2.0f
        / (3.0f * float(scene.dimensions.x) * float(scene.dimensions.y));
    float3 lossGradient = normalization * (finalColor - target[pixelID].xyz);
    float3 ray = cameraRay(safePixel, scene);
    float3 prefixColor = float3(0.0f);
    float transmittance = 1.0f;
    uint tile = tileForPixel(safePixel, scene);

    for (uint entry = tileOffsets[tile]; entry < tileOffsets[tile + 1]; ++entry) {
        if (simd_all(!active)) {
            break;
        }
        uint primitiveID = tilePrimitiveIDs[entry];
        uint base = primitiveID * kParameterStride;
        float localGradient[20] = {0.0f};
        HexelSample sample;
        bool hit = active && evaluateHexel(parameters, primitiveID, ray, scene, sample);
        if (hit && transmittance * (1.0f - sample.alpha) < kTransmittanceCutoff) {
            active = false;
            hit = false;
        }
        if (hit) {
            float remaining = transmittance * (1.0f - sample.alpha);
            device const float *row = parameters + base;
            float weight = transmittance * sample.alpha;
            float3 nextPrefix = prefixColor + weight * sample.color;
            float3 colorGradient = weight * lossGradient;
            // C_after is obtained from the stored final composite. This expression
            // avoids division by tiny transmittance: 1-alpha is at least 0.01.
            float alphaGradient = dot(lossGradient,
                transmittance * sample.color
                - (finalColor - nextPrefix) / (1.0f - sample.alpha));

            storeGradient3(localGradient, 9, colorGradient);
            storeGradient3(localGradient, 12, sample.u * colorGradient);
            storeGradient3(localGradient, 15, sample.v * colorGradient);

            float distanceGradient = 0.0f;
            float logSoftnessGradient = 0.0f;
            float opacityGradient = 0.0f;
            if (sample.rawAlpha <= kAlphaMaximum) {
                float boundaryFactor = alphaGradient * sample.opacity
                    * sample.boundary * (1.0f - sample.boundary);
                distanceGradient = -sample.softness * boundaryFactor;
                if (sample.rawSoftness > kMinimumSoftness) {
                    logSoftnessGradient = -sample.distance * sample.rawSoftness
                        * boundaryFactor;
                }
                opacityGradient = alphaGradient * sample.boundary
                    * sample.opacity * (1.0f - sample.opacity);
            }
            localGradient[18] = opacityGradient;
            localGradient[19] = logSoftnessGradient;

            float uGradient = dot(colorGradient, float3(row[12], row[13], row[14]))
                + distanceGradient * sample.distanceGradient.x;
            float vGradient = dot(colorGradient, float3(row[15], row[16], row[17]))
                + distanceGradient * sample.distanceGradient.y;
            localGradient[3] = -uGradient * sample.u;
            localGradient[4] = -vGradient * sample.v;

            float3 axisUGradient = uGradient * sample.inverseScale.x * sample.intersection;
            float3 axisVGradient = vGradient * sample.inverseScale.y * sample.intersection;
            float3 intersectionGradient = uGradient * sample.inverseScale.x * sample.axisU
                + vGradient * sample.inverseScale.y * sample.axisV;
            float depthGradient = dot(intersectionGradient, ray);
            float numeratorGradient = depthGradient / sample.denominator;
            float denominatorGradient = -depthGradient * sample.depth / sample.denominator;
            float3 normalGradient = numeratorGradient * sample.relativeCenter
                + denominatorGradient * ray;
            float3 centerGradient = -intersectionGradient
                + numeratorGradient * sample.normal;
            storeGradient3(localGradient, 0, centerGradient);
            float4 rotationGradient = quaternionVJP(sample, axisUGradient,
                                                   axisVGradient, normalGradient);
            for (uint component = 0; component < 4; ++component) {
                localGradient[5 + component] = rotationGradient[component];
            }
            prefixColor = nextPrefix;
            transmittance = remaining;
        }
        // Never put these reductions inside the per-pixel hit branch: every
        // active SIMD lane must enter every reduction with a value (or zero).
        // One lane performs the global update instead of every covered pixel.
        for (uint component = 0; component < kParameterStride; ++component) {
            float total = simd_sum(localGradient[component]);
            if (lane == 0 && total != 0.0f) {
                atomic_fetch_add_explicit(gradient + base + component, total,
                                          memory_order_relaxed);
            }
        }
    }
}

kernel void adamUpdate(device float *parameters [[buffer(0)]],
                       device const float *gradient [[buffer(1)]],
                       device float *firstMoment [[buffer(2)]],
                       device float *secondMoment [[buffer(3)]],
                       constant AdamUniforms &adam [[buffer(4)]],
                       device const float *learningRates [[buffer(5)]],
                       uint parameterID [[thread_position_in_grid]]) {
    if (parameterID >= adam.parameterCount) {
        return;
    }
    float rate = learningRates[parameterID % kParameterStride];
    // A zero learning rate freezes both the parameter and its optimizer state.
    if (rate == 0.0f) {
        return;
    }
    float value = gradient[parameterID];
    float first = adam.beta1 * firstMoment[parameterID] + (1.0f - adam.beta1) * value;
    float second = adam.beta2 * secondMoment[parameterID]
        + (1.0f - adam.beta2) * value * value;
    firstMoment[parameterID] = first;
    secondMoment[parameterID] = second;
    float correctedFirst = first / (1.0f - adam.beta1Power);
    float correctedSecond = second / (1.0f - adam.beta2Power);
    parameters[parameterID] -= rate * correctedFirst
        / (sqrt(correctedSecond) + adam.epsilon);
}
