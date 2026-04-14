// ============================================================================
// copywrite SynthDef Library
// ============================================================================
//
// Effects:
//   \sp1200         - Bit crusher / sample rate reducer (E-MU SP-1200 style)
//                     Params: in, out, bitDepth=12, sampleRate=26040, mix=1.0
//   \alesis3630     - Compressor (Alesis 3630 style)
//                     Params: in, out, threshold=0.5, ratio=4, attack=0.01,
//                             release=0.1, makeupGain=1.0
//   \filterSweep    - Resonant multi-mode filter with LFO
//                     Params: in, out, cutoff=1000, resonance=0.5,
//                             filterType=0 (0=LP,1=HP,2=BP), lfoRate=0.1,
//                             lfoDepth=0, envAmount=0
//   \vocoder        - 16-band channel vocoder
//                     Params: modulator, carrier, out, numBands=16,
//                             bandwidth=0.5
//   \sidechain      - Amplitude ducker
//                     Params: in, trigIn, out, depth=0.8, attack=0.005,
//                             release=0.1
//   \phaserFlanger  - All-pass chain phaser/flanger
//                     Params: in, out, rate=0.25, depth=0.5, feedback=0.5,
//                             mix=0.5
//
// Drums:
//   \kick           - Synthesized kick drum
//                     Params: out, freq=50, click=0.5, decay=0.5, amp=0.8,
//                             pan=0
//   \snare          - Synthesized snare drum
//                     Params: out, freq=200, noiseAmt=0.5, decay=0.2,
//                             amp=0.6, pan=0
//   \hihat          - Synthesized hihat
//                     Params: out, decay=0.05, hpFreq=8000, amp=0.3, pan=0
//   \clap           - Synthesized clap
//                     Params: out, decay=0.15, amp=0.5, pan=0
//
// Instruments:
//   \miniVoyager    - 3-oscillator synth with Moog ladder filter
//                     Params: out, freq=440, gate=1, detune=0.01,
//                             filterCutoff=2000, filterRes=0.3, attack=0.01,
//                             decay=0.3, sustain=0.7, release=0.5, amp=0.5,
//                             pan=0
//   \fenderRhodes   - FM electric piano
//                     Params: out, freq=440, gate=1, modIndex=2, modRatio=1,
//                             attack=0.01, decay=0.5, sustain=0.3,
//                             release=0.5, amp=0.5, pan=0
//   \bassline       - Distorted/saturated bass synth
//                     Params: out, freq=110, gate=1, filterCutoff=800,
//                             filterRes=0.6, drive=2.0, attack=0.01,
//                             decay=0.2, sustain=0.8, release=0.1, amp=0.6,
//                             pan=0
//   \padSynth       - Detuned saw/square pad with filter
//                     Params: out, freq=440, gate=1, detune=0.02,
//                             mix=0.5 (saw/square), filterCutoff=3000,
//                             filterRes=0.3, attack=0.5, decay=1.0,
//                             sustain=0.7, release=2.0, amp=0.4, pan=0
//   \crowdNoise     - Shaped noise for stadium atmosphere
//                     Params: out, density=0.5, hpFreq=200, lpFreq=4000,
//                             amp=0.3, pan=0
//
// ============================================================================

(
var defDir = Platform.userAppSupportDir +/+ "synthdefs";
File.mkdir(defDir);

// ---- Effects ----

SynthDef(\sp1200, { |in=0, out=0, bitDepth=12, sampleRate=26040, mix=1.0|
	var sig, dry, crushed, steps, downsampled;
	dry = In.ar(in, 2);
	steps = (2 ** bitDepth) * 0.5;
	downsampled = Latch.ar(dry, Impulse.ar(sampleRate));
	crushed = (downsampled * steps).round / steps;
	sig = XFade2.ar(dry, crushed, mix * 2 - 1);
	Out.ar(out, sig);
}).writeDefFile(defDir);

SynthDef(\alesis3630, { |in=0, out=0, threshold=0.5, ratio=4, attack=0.01, release=0.1, makeupGain=1.0|
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

SynthDef(\filterSweep, { |in=0, out=0, cutoff=1000, resonance=0.5, filterType=0, lfoRate=0.1, lfoDepth=0, envAmount=0|
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

SynthDef(\vocoder, { |modulator=0, carrier=2, out=0, numBands=16, bandwidth=0.5|
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

SynthDef(\sidechain, { |in=0, trigIn=2, out=0, depth=0.8, attack=0.005, release=0.1|
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

SynthDef(\phaserFlanger, { |in=0, out=0, rate=0.25, depth=0.5, feedback=0.5, mix=0.5|
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

// ---- Drums ----

SynthDef(\kick, { |out=0, freq=50, click=0.5, decay=0.5, amp=0.8, pan=0|
	var freqEnv, sig, clickSig, env;
	freqEnv = EnvGen.kr(Env.perc(0.001, 0.07), doneAction: 0) * freq * 6;
	sig = SinOsc.ar(freq + freqEnv);
	clickSig = WhiteNoise.ar * EnvGen.kr(Env.perc(0.001, 0.01)) * click;
	sig = sig + clickSig;
	env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);
	sig = (sig * env * amp).tanh;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\snare, { |out=0, freq=200, noiseAmt=0.5, decay=0.2, amp=0.6, pan=0|
	var tone, noise, sig, env;
	tone = SinOsc.ar(freq) * EnvGen.kr(Env.perc(0.001, 0.05));
	noise = WhiteNoise.ar * EnvGen.kr(Env.perc(0.001, decay)) * noiseAmt;
	sig = tone + noise;
	env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);
	sig = HPF.ar(sig, 100);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\hihat, { |out=0, decay=0.05, hpFreq=8000, amp=0.3, pan=0|
	var sig, env;
	sig = WhiteNoise.ar + (Mix.ar(Pulse.ar([800, 1557, 2093, 3175, 4200, 5543], 0.5, 0.05)));
	sig = HPF.ar(sig, hpFreq);
	env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\clap, { |out=0, decay=0.15, amp=0.5, pan=0|
	var sig, env, noise;
	noise = BPF.ar(WhiteNoise.ar, 1200, 0.5);
	sig = Mix.ar(Array.fill(4, { |i|
		DelayN.ar(noise * EnvGen.kr(Env.perc(0.001, 0.005)), 0.05, i * 0.008)
	}));
	env = EnvGen.kr(Env.perc(0.005, decay), doneAction: 2);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

// ---- Instruments ----

SynthDef(\miniVoyager, { |out=0, freq=440, gate=1, detune=0.01, filterCutoff=2000, filterRes=0.3, attack=0.01, decay=0.3, sustain=0.7, release=0.5, amp=0.5, pan=0|
	var osc1, osc2, osc3, sig, env, fenv;
	osc1 = Saw.ar(freq);
	osc2 = Saw.ar(freq * (1 + detune));
	osc3 = Pulse.ar(freq * (1 - detune), 0.5);
	sig = Mix.ar([osc1, osc2, osc3]) * 0.33;
	fenv = EnvGen.kr(Env.adsr(attack, decay * 0.5, 0.5, release), gate) * filterCutoff;
	sig = MoogFF.ar(sig, (filterCutoff + fenv).clip(20, 20000), filterRes * 4);
	env = EnvGen.kr(Env.adsr(attack, decay, sustain, release), gate, doneAction: 2);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\fenderRhodes, { |out=0, freq=440, gate=1, modIndex=2, modRatio=1, attack=0.01, decay=0.5, sustain=0.3, release=0.5, amp=0.5, pan=0|
	var modFreq, modulator, carrier, sig, env, tremolo;
	modFreq = freq * modRatio;
	modulator = SinOsc.ar(modFreq) * modFreq * modIndex;
	carrier = SinOsc.ar(freq + modulator);
	env = EnvGen.kr(Env.adsr(attack, decay, sustain, release), gate, doneAction: 2);
	tremolo = SinOsc.kr(5.2).range(0.9, 1);
	sig = carrier * env * tremolo * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\bassline, { |out=0, freq=110, gate=1, filterCutoff=800, filterRes=0.6, drive=2.0, attack=0.01, decay=0.2, sustain=0.8, release=0.1, amp=0.6, pan=0|
	var osc, sub, sig, env, fenv;
	osc = Saw.ar(freq) + Pulse.ar(freq, 0.3);
	sub = SinOsc.ar(freq * 0.5) * 0.5;
	sig = Mix.ar([osc, sub]) * 0.5;
	sig = (sig * drive).tanh;
	fenv = EnvGen.kr(Env.adsr(attack, decay * 0.3, 0.3, release * 0.5), gate) * filterCutoff;
	sig = RLPF.ar(sig, (filterCutoff + fenv).clip(20, 20000), (1 - filterRes).clip(0.01, 1));
	env = EnvGen.kr(Env.adsr(attack, decay, sustain, release), gate, doneAction: 2);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\padSynth, { |out=0, freq=440, gate=1, detune=0.02, mix=0.5, filterCutoff=3000, filterRes=0.3, attack=0.5, decay=1.0, sustain=0.7, release=2.0, amp=0.4, pan=0|
	var saws, squares, sig, env;
	saws = Mix.ar(Array.fill(4, { |i|
		Saw.ar(freq * (1 + (detune * Rand(-1.0, 1.0))))
	})) * 0.25;
	squares = Mix.ar(Array.fill(4, { |i|
		Pulse.ar(freq * (1 + (detune * Rand(-1.0, 1.0))), 0.5)
	})) * 0.25;
	sig = (saws * (1 - mix)) + (squares * mix);
	sig = RLPF.ar(sig, filterCutoff, (1 - filterRes).clip(0.01, 1));
	env = EnvGen.kr(Env.adsr(attack, decay, sustain, release), gate, doneAction: 2);
	sig = sig * env * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

SynthDef(\crowdNoise, { |out=0, density=0.5, hpFreq=200, lpFreq=4000, amp=0.3, pan=0|
	var noise1, noise2, rumble, sig;
	noise1 = PinkNoise.ar * density;
	noise2 = BrownNoise.ar * (1 - density) * 0.5;
	rumble = LFNoise1.ar(0.5) * SinOsc.ar(LFNoise0.kr(0.2).range(60, 150)) * 0.1 * density;
	sig = noise1 + noise2 + rumble;
	sig = HPF.ar(sig, hpFreq);
	sig = LPF.ar(sig, lpFreq);
	sig = sig * LFNoise2.kr(0.3).range(0.7, 1.0);
	sig = sig * amp;
	Out.ar(out, Pan2.ar(sig, pan));
}).writeDefFile(defDir);

"copywrite SynthDef library loaded.".postln;
0.exit;
)
