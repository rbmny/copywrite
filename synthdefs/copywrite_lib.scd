// ============================================================================
// copywrite SynthDef Library — Daft Punk "Homework" era
// ============================================================================
//
// Modeled after: TR-909, TB-303, Juno-106, Alesis 3630, SP-1200
//
// Signal chain (per the magazine scan):
//   Sound Source → Effects (distortion, filter) → Mixer → Alesis 3630 → DAT
//
// ============================================================================

(
var defDir = Platform.userAppSupportDir +/+ "synthdefs";
File.mkdir(defDir);

// ---- DRUMS: Roland TR-909 style ----

// 909 Kick — pitch sweep from high to low, long body, punchy
SynthDef(\kick, { |out=0, freq=52, punch=1.0, decay=0.7, amp=0.8, pan=0|
    var pitchEnv, body, click, env, sig;
    // 909 kick: pitch sweeps down from ~4x to fundamental
    pitchEnv = EnvGen.kr(Env([freq * 4, freq * 1.5, freq], [0.005, 0.06], [-8, -4]));
    body = SinOsc.ar(pitchEnv);
    // Transient click for attack
    click = LPF.ar(WhiteNoise.ar, 8000) * EnvGen.kr(Env.perc(0.0005, 0.003)) * punch;
    sig = body + click;
    // Long, smooth decay — 909 kicks ring
    env = EnvGen.kr(Env.perc(0.001, decay, curve: -6), doneAction: 2);
    sig = (sig * env * amp * 1.5).tanh;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// 909 Snare — pitched body + noise burst, longer tail
SynthDef(\snare, { |out=0, freq=185, tone=0.5, decay=0.25, amp=0.6, pan=0|
    var body, noise, sig, env;
    body = SinOsc.ar(freq) * EnvGen.kr(Env.perc(0.0005, 0.08, curve: -6));
    body = body + (SinOsc.ar(freq * 1.6) * EnvGen.kr(Env.perc(0.0005, 0.04)) * 0.5);
    noise = HPF.ar(WhiteNoise.ar, 1800) * EnvGen.kr(Env.perc(0.001, decay, curve: -4));
    sig = (body * tone) + (noise * (1 - tone * 0.3));
    env = EnvGen.kr(Env.perc(0.001, decay * 1.5), doneAction: 2);
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// 909 Closed Hihat — metallic, tight
SynthDef(\hihat, { |out=0, decay=0.06, hpFreq=7000, amp=0.3, pan=0|
    var sig, env;
    // 909 hihats use 6 square wave oscillators at non-harmonic ratios
    sig = Mix.ar([
        Pulse.ar(800, 0.5), Pulse.ar(1557, 0.5), Pulse.ar(2093, 0.5),
        Pulse.ar(3175, 0.5), Pulse.ar(4200, 0.5), Pulse.ar(5543, 0.5)
    ]) * 0.1;
    sig = sig + (WhiteNoise.ar * 0.3);
    sig = HPF.ar(sig, hpFreq);
    sig = BPF.ar(sig, 10000, 0.5) + (sig * 0.3);
    env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// 909 Open Hihat
SynthDef(\openhat, { |out=0, decay=0.35, hpFreq=7000, amp=0.25, pan=0|
    var sig, env;
    sig = Mix.ar([
        Pulse.ar(800, 0.5), Pulse.ar(1557, 0.5), Pulse.ar(2093, 0.5),
        Pulse.ar(3175, 0.5), Pulse.ar(4200, 0.5), Pulse.ar(5543, 0.5)
    ]) * 0.1;
    sig = sig + (WhiteNoise.ar * 0.3);
    sig = HPF.ar(sig, hpFreq);
    env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// 909 Clap — multiple short noise bursts then reverb tail
SynthDef(\clap, { |out=0, decay=0.2, amp=0.5, pan=0|
    var sig, env, noise;
    noise = BPF.ar(WhiteNoise.ar, 1100, 0.7);
    // 909 clap: 4 rapid bursts then a longer tail
    sig = Mix.ar(Array.fill(4, { |i|
        DelayN.ar(noise * EnvGen.kr(Env.perc(0.001, 0.006)), 0.05, i * 0.01)
    }));
    env = EnvGen.kr(Env.perc(0.005, decay), doneAction: 2);
    sig = sig * env * amp;
    // Add some room reverb (909 claps had built-in reverb)
    sig = sig + (FreeVerb.ar(sig, 0.4, 0.3, 0.5) * 0.3);
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// ---- BASS: Roland TB-303 style ----

// TB-303 Bassline — THE Da Funk sound
// Saw/Square through resonant diode ladder filter, heavy distortion
SynthDef(\bassline, { |out=0, freq=110, gate=1,
    filterCutoff=800, filterRes=0.7, filterEnv=2000,
    waveform=0, // 0=saw, 1=square
    accent=0.5, slide=0, drive=3.0,
    attack=0.005, decay=0.15, sustain=0.6, release=0.4, amp=0.5, pan=0|
    var osc, sig, env, fenv, cutoff;
    // 303: single oscillator, saw or square
    osc = Select.ar(waveform, [
        LFSaw.ar(freq),
        LFPulse.ar(freq, width: 0.5) * 2 - 1
    ]);
    // Sub oscillator for extra weight
    osc = osc + (SinOsc.ar(freq * 0.5) * 0.4);
    // Filter envelope — the 303's "accent" feature
    fenv = EnvGen.kr(Env.perc(0.001, decay * 0.5, curve: -8)) * filterEnv * accent;
    cutoff = (filterCutoff + fenv).clip(30, 18000);
    // Diode ladder filter emulation (MoogFF is close enough)
    sig = MoogFF.ar(osc, cutoff, filterRes * 3.5);
    // Heavy distortion — this is what makes Daft Punk sound dirty
    sig = (sig * drive).tanh;
    sig = (sig * drive * 0.7).softclip;
    // Amplitude envelope with longer release (no abrupt death)
    env = EnvGen.kr(
        Env.adsr(attack, decay, sustain, release),
        gate, doneAction: 2
    );
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// ---- SYNTHS: Juno-106 style ----

// Juno Pad — detuned saw/pulse, chorus, warm filter
SynthDef(\padSynth, { |out=0, freq=440, gate=1,
    detune=0.008, mix=0.5,
    filterCutoff=2000, filterRes=0.2,
    attack=0.4, decay=0.8, sustain=0.7, release=1.5,
    amp=0.3, pan=0|
    var saws, pulses, sig, env, chorus;
    // Juno-style: slightly detuned oscillators
    saws = Mix.ar(Array.fill(3, { |i|
        LFSaw.ar(freq * (1 + (detune * (i - 1))))
    })) * 0.33;
    pulses = Mix.ar(Array.fill(3, { |i|
        LFPulse.ar(freq * (1 + (detune * (i - 1))), width: 0.5) * 2 - 1
    })) * 0.33;
    sig = (saws * (1 - mix)) + (pulses * mix);
    // Juno's built-in chorus
    chorus = DelayC.ar(sig, 0.03, SinOsc.kr(0.8).range(0.003, 0.012));
    sig = sig + (chorus * 0.5);
    // Warm lowpass filter
    sig = RLPF.ar(sig, filterCutoff.clip(30, 18000), (1 - filterRes).clip(0.05, 1));
    env = EnvGen.kr(
        Env.adsr(attack, decay, sustain, release),
        gate, doneAction: 2
    );
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// Lead synth — Moog-style, 3 osc, ladder filter
SynthDef(\miniVoyager, { |out=0, freq=440, gate=1,
    detune=0.01, filterCutoff=3000, filterRes=0.3,
    attack=0.01, decay=0.3, sustain=0.7, release=0.8,
    drive=1.5, amp=0.4, pan=0|
    var osc1, osc2, osc3, sig, env, fenv;
    osc1 = LFSaw.ar(freq);
    osc2 = LFSaw.ar(freq * (1 + detune));
    osc3 = LFPulse.ar(freq * (1 - detune), width: 0.5) * 2 - 1;
    sig = Mix.ar([osc1, osc2, osc3]) * 0.33;
    fenv = EnvGen.kr(Env.perc(0.001, decay * 0.4, curve: -6)) * filterCutoff * 0.5;
    sig = MoogFF.ar(sig, (filterCutoff + fenv).clip(30, 18000), filterRes * 3.5);
    sig = (sig * drive).tanh;
    env = EnvGen.kr(
        Env.adsr(attack, decay, sustain, release),
        gate, doneAction: 2
    );
    sig = sig * env * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// FM Rhodes — electric piano
SynthDef(\fenderRhodes, { |out=0, freq=440, gate=1,
    modIndex=2, modRatio=1,
    attack=0.01, decay=0.8, sustain=0.3, release=0.8,
    amp=0.4, pan=0|
    var modFreq, modulator, carrier, sig, env, tremolo;
    modFreq = freq * modRatio;
    modulator = SinOsc.ar(modFreq) * modFreq * modIndex;
    carrier = SinOsc.ar(freq + modulator);
    env = EnvGen.kr(
        Env.adsr(attack, decay, sustain, release),
        gate, doneAction: 2
    );
    tremolo = SinOsc.kr(5.2).range(0.85, 1);
    sig = carrier * env * tremolo * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// ---- EFFECTS ----

// SP-1200 style bit crusher
SynthDef(\sp1200, { |in=0, out=0, bitDepth=12, sampleRate=26040, mix=1.0|
    var sig, dry, crushed, steps, downsampled;
    dry = In.ar(in, 2);
    steps = (2 ** bitDepth) * 0.5;
    downsampled = Latch.ar(dry, Impulse.ar(sampleRate));
    crushed = (downsampled * steps).round / steps;
    sig = XFade2.ar(dry, crushed, mix * 2 - 1);
    Out.ar(out, sig);
}).writeDefFile(defDir);

// Alesis 3630 Compressor
SynthDef(\alesis3630, { |in=0, out=0,
    threshold=0.5, ratio=4, attack=0.01, release=0.1, makeupGain=1.0|
    var sig, compressed;
    sig = In.ar(in, 2);
    compressed = Compander.ar(
        sig, sig,
        thresh: threshold,
        slopeBelow: 1,
        slopeAbove: ratio.reciprocal,
        clampTime: attack,
        relaxTime: release
    );
    Out.ar(out, compressed * makeupGain);
}).writeDefFile(defDir);

// Resonant filter sweep
SynthDef(\filterSweep, { |in=0, out=0,
    cutoff=1000, resonance=0.5, filterType=0,
    lfoRate=0.1, lfoDepth=0, envAmount=0|
    var sig, lfo, modCutoff, filtered, lp, hp, bp, rq;
    sig = In.ar(in, 2);
    lfo = SinOsc.kr(lfoRate).range(0, 1) * lfoDepth;
    modCutoff = (cutoff + (lfo * cutoff) + (envAmount * cutoff)).clip(20, 20000);
    rq = (1 - resonance).clip(0.01, 1);
    lp = RLPF.ar(sig, modCutoff, rq);
    hp = RHPF.ar(sig, modCutoff, rq);
    bp = BPF.ar(sig, modCutoff, rq);
    filtered = Select.ar(filterType, [lp, hp, bp]);
    Out.ar(out, filtered);
}).writeDefFile(defDir);

// Sidechain ducker
SynthDef(\sidechain, { |in=0, trigIn=2, out=0,
    depth=0.8, attack=0.005, release=0.1|
    var sig, trig, env;
    sig = In.ar(in, 2);
    trig = In.ar(trigIn, 1);
    env = Compander.ar(
        sig, trig,
        thresh: 0.1,
        slopeBelow: 1,
        slopeAbove: 1 - depth,
        clampTime: attack,
        relaxTime: release
    );
    Out.ar(out, env);
}).writeDefFile(defDir);

// Phaser/Flanger
SynthDef(\phaserFlanger, { |in=0, out=0,
    rate=0.25, depth=0.5, feedback=0.5, mix=0.5|
    var sig, dry, wet, mod, chain;
    dry = In.ar(in, 2);
    mod = SinOsc.kr(rate).range(0.0001, 0.01) * depth;
    chain = dry;
    6.do { |i|
        chain = AllpassC.ar(chain, 0.02, mod * ((i + 1) / 6), 0);
    };
    wet = chain + (LocalIn.ar(2) * feedback);
    LocalOut.ar(wet);
    sig = XFade2.ar(dry, wet, mix * 2 - 1);
    Out.ar(out, sig);
}).writeDefFile(defDir);

// Vocoder
SynthDef(\vocoder, { |modulator=0, carrier=2, out=0,
    numBands=16, bandwidth=0.5|
    var modSig, carSig, bands, mixed;
    modSig = In.ar(modulator, 1);
    carSig = In.ar(carrier, 1);
    bands = Array.fill(16, { |i|
        var freq = 80 * (2 ** (i * (10.0 / 16)));
        var bw = freq * bandwidth.clip(0.1, 2.0);
        var modBand = BPF.ar(modSig, freq, bw / freq);
        var carBand = BPF.ar(carSig, freq, bw / freq);
        var follower = Amplitude.kr(modBand, 0.01, 0.05);
        carBand * follower;
    });
    mixed = Mix.ar(bands) * 4;
    Out.ar(out, Pan2.ar(mixed));
}).writeDefFile(defDir);

// Silent keeper — for NRT to not stop early
SynthDef(\crowdNoise, { |out=0, density=0.5, hpFreq=200, lpFreq=4000, amp=0.3, pan=0|
    var noise1, noise2, sig;
    noise1 = PinkNoise.ar * density;
    noise2 = BrownNoise.ar * (1 - density) * 0.5;
    sig = noise1 + noise2;
    sig = HPF.ar(sig, hpFreq);
    sig = LPF.ar(sig, lpFreq);
    sig = sig * LFNoise2.kr(0.3).range(0.7, 1.0) * amp;
    Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

"copywrite SynthDef library loaded.".postln;
0.exit;
)
