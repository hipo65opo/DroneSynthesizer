use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use std::collections::VecDeque;
use rand::Rng;

const SAMPLE_RATE: f32 = 44100.0;
const TWO_PI: f32 = std::f32::consts::PI * 2.0;
const GRAIN_BUFFER_SIZE: usize = 44100 * 2; // 2秒
const MAX_DELAY_SAMPLES: usize = 44100 * 2; // 2秒
const LFO_FREQUENCY: f32 = 0.1; // 0.1Hz = 10秒周期
const STEPS: usize = 16;
const DEFAULT_BPM: f32 = 120.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum SynthType {
    HarmonicDrone,
    FMSynth,
    SubtractiveSynth,
    GranularDrone,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Waveform {
    Sine,
    Square,
    Triangle,
    Sawtooth,
}

impl Waveform {
    fn generate_sample(&self, phase: f32) -> f32 {
        match self {
            Waveform::Sine => (phase * TWO_PI).sin(),
            Waveform::Square => if phase < 0.5 { 1.0 } else { -1.0 },
            Waveform::Triangle => {
                if phase < 0.25 {
                    phase * 4.0
                } else if phase < 0.75 {
                    2.0 - (phase * 4.0)
                } else {
                    (phase * 4.0) - 4.0
                }
            },
            Waveform::Sawtooth => 2.0 * phase - 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FilterMode {
    LowPass,
    HighPass,
    BandPass,
}

#[derive(Debug)]
struct Filter {
    mode: FilterMode,
    cutoff: f32,
    resonance: f32,
    last_input: f32,
    last_output: f32,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            mode: FilterMode::LowPass,
            cutoff: 1000.0,
            resonance: 0.5,
            last_input: 0.0,
            last_output: 0.0,
        }
    }
}

impl Filter {
    fn process(&mut self, input: f32) -> f32 {
        let normalized_cutoff = (2.0 * std::f32::consts::PI * self.cutoff / SAMPLE_RATE).min(1.0);
        let q = self.resonance.max(0.01);
        
        let alpha = normalized_cutoff / (2.0 * q);
        let cosw0 = (normalized_cutoff / 2.0).cos();
        
        let a0 = 1.0 + alpha;
        let b0 = match self.mode {
            FilterMode::LowPass => (1.0 - cosw0) / 2.0,
            FilterMode::HighPass => (1.0 + cosw0) / 2.0,
            FilterMode::BandPass => alpha,
        };
        let b1 = match self.mode {
            FilterMode::LowPass => 1.0 - cosw0,
            FilterMode::HighPass => -(1.0 + cosw0),
            FilterMode::BandPass => 0.0,
        };
        let b2 = match self.mode {
            FilterMode::LowPass => (1.0 - cosw0) / 2.0,
            FilterMode::HighPass => (1.0 + cosw0) / 2.0,
            FilterMode::BandPass => -alpha,
        };

        let output = (b0 * input + b1 * self.last_input + b2 * self.last_output) / a0;
        self.last_input = input;
        self.last_output = output;
        
        output
    }
}

#[derive(Debug)]
struct Delay {
    buffer: VecDeque<f32>,
    delay_time: f32,
    feedback: f32,
}

impl Default for Delay {
    fn default() -> Self {
        Self {
            buffer: VecDeque::with_capacity(MAX_DELAY_SAMPLES),
            delay_time: 0.5,
            feedback: 0.5,
        }
    }
}

impl Delay {
    fn process(&mut self, input: f32) -> f32 {
        if self.delay_time <= 0.001 {
            return input;
        }

        self.buffer.push_back(input);
        if self.buffer.len() > MAX_DELAY_SAMPLES {
            self.buffer.pop_front();
        }

        let delay_samples = (self.delay_time * SAMPLE_RATE).max(1.0) as usize;
        let delayed = if self.buffer.len() > delay_samples {
            self.buffer[self.buffer.len() - delay_samples]
        } else {
            0.0
        };

        input + delayed * self.feedback
    }
}

#[derive(Debug)]
struct Reverb {
    buffers: Vec<VecDeque<f32>>,
    decay: f32,
    room_size: f32,
    wet_mix: f32,
}

impl Default for Reverb {
    fn default() -> Self {
        // 異なるディレイ時間を持つ複数のバッファを作成
        let delay_times = [29, 37, 43, 47, 53, 59, 61, 67];
        let buffers = delay_times.iter()
            .map(|&size| VecDeque::with_capacity((size * 1000) as usize))
            .collect();

        Self {
            buffers,
            decay: 0.5,
            room_size: 0.5,
            wet_mix: 0.3,
        }
    }
}

impl Reverb {
    fn process(&mut self, input: f32) -> f32 {
        let mut output = 0.0;
        let room_scale = self.room_size * 1000.0;

        for buffer in &mut self.buffers {
            buffer.push_back(input);
            if buffer.len() > (room_scale as usize) {
                if let Some(delayed) = buffer.pop_front() {
                    output += delayed * self.decay;
                }
            }
        }

        input * (1.0 - self.wet_mix) + output * self.wet_mix
    }
}

#[derive(Debug)]
struct LFO {
    phase: f32,
    frequency: f32,
    amount: f32,
    target: LFOTarget,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LFOTarget {
    Frequency,
    Volume,
    FilterCutoff,
    FilterResonance,
}

impl Default for LFO {
    fn default() -> Self {
        Self {
            phase: 0.0,
            frequency: 0.1,
            amount: 0.5,
            target: LFOTarget::Frequency,
        }
    }
}

impl LFO {
    fn next_value(&mut self) -> f32 {
        let value = (self.phase * TWO_PI).sin() * self.amount;
        self.phase += self.frequency / SAMPLE_RATE;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum EnvelopeTarget {
    Amplitude,
    FilterCutoff,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Debug)]
struct Envelope {
    attack_time: f32,
    decay_time: f32,
    sustain_level: f32,
    release_time: f32,
    current_level: f32,
    state: EnvelopeState,
    start_time: std::time::Instant,
    start_level: f32,
    target: EnvelopeTarget,
}

impl Default for Envelope {
    fn default() -> Self {
        Self {
            attack_time: 0.1,
            decay_time: 0.2,
            sustain_level: 0.7,
            release_time: 0.3,
            current_level: 0.0,
            state: EnvelopeState::Sustain,
            start_time: std::time::Instant::now(),
            start_level: 0.0,
            target: EnvelopeTarget::Amplitude,
        }
    }
}

impl Envelope {
    fn next_value(&mut self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        self.current_level = match self.state {
            EnvelopeState::Idle => 0.0,
            EnvelopeState::Attack => {
                if elapsed >= self.attack_time {
                    self.state = EnvelopeState::Decay;
                    self.start_level = 1.0;
                    self.start_time = std::time::Instant::now();
                    1.0
                } else {
                    self.start_level + (1.0 - self.start_level) * (elapsed / self.attack_time)
                }
            },
            EnvelopeState::Decay => {
                if elapsed >= self.decay_time {
                    self.state = EnvelopeState::Sustain;
                    self.sustain_level
                } else {
                    1.0 - (1.0 - self.sustain_level) * (elapsed / self.decay_time)
                }
            },
            EnvelopeState::Sustain => self.sustain_level,
            EnvelopeState::Release => {
                if elapsed >= self.release_time {
                    self.state = EnvelopeState::Idle;
                    0.0
                } else {
                    self.start_level * (1.0 - elapsed / self.release_time)
                }
            },
        };

        self.current_level
    }

    fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
        self.start_level = self.current_level;
        self.start_time = std::time::Instant::now();
    }

    fn release(&mut self) {
        if self.state != EnvelopeState::Idle {
            self.state = EnvelopeState::Release;
            self.start_level = self.current_level;
            self.start_time = std::time::Instant::now();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArpPattern {
    Up,
    Down,
    UpDown,
    Random,
}

#[derive(Debug)]
struct Arpeggiator {
    enabled: bool,
    pattern: ArpPattern,
    rate: f32,  // Hz
    octave_range: i32,
    current_step: usize,
    current_phase: f32,
    base_note: i32,
    notes: Vec<i32>,
    rng: rand::rngs::SmallRng,
}

impl Default for Arpeggiator {
    fn default() -> Self {
        Self {
            enabled: false,
            pattern: ArpPattern::Up,
            rate: 4.0,  // 4Hz = 16th notes at 60 BPM
            octave_range: 1,
            current_step: 0,
            current_phase: 0.0,
            base_note: 60,  // Middle C
            notes: Vec::new(),
            rng: rand::SeedableRng::from_entropy(),
        }
    }
}

impl Arpeggiator {
    fn trigger(&mut self, note: i32) {
        self.base_note = note;
        self.current_step = 0;  // ステップをリセット
        self.current_phase = 0.0;  // フェーズをリセット
        self.update_notes();
    }

    fn update_notes(&mut self) {
        self.notes.clear();
        let base_notes = [0, 4, 7]; // メジャートライアド
        
        for octave in 0..=self.octave_range {
            for &interval in &base_notes {
                self.notes.push(self.base_note + interval + (octave * 12));
            }
        }

        match self.pattern {
            ArpPattern::Down => self.notes.reverse(),
            ArpPattern::UpDown => {
                let mut down_notes = self.notes[1..self.notes.len()-1].to_vec();
                down_notes.reverse();
                self.notes.extend(down_notes);
            },
            ArpPattern::Random => {
                use rand::seq::SliceRandom;
                self.notes.shuffle(&mut self.rng);
            },
            _ => {}
        }
    }

    fn next_note(&mut self) -> Option<i32> {
        if !self.enabled || self.notes.is_empty() {
            return None;
        }

        self.current_phase += self.rate / SAMPLE_RATE;
        if self.current_phase >= 1.0 {
            self.current_phase -= 1.0;
            self.current_step = (self.current_step + 1) % self.notes.len();
            Some(self.notes[self.current_step])
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Flanger {
    buffer: VecDeque<f32>,
    lfo_phase: f32,
    rate: f32,
    depth: f32,
    feedback: f32,
    wet_mix: f32,
}

impl Default for Flanger {
    fn default() -> Self {
        let mut buffer = VecDeque::with_capacity(4410);
        // バッファを0.0で初期化
        for _ in 0..4410 {
            buffer.push_back(0.0);
        }
        Self {
            buffer,
            lfo_phase: 0.0,
            rate: 0.5,     // 0.5に設定
            depth: 0.5,    // 0.5に設定
            feedback: 0.5,  // 0.5に設定
            wet_mix: 0.5,  // 0.5に設定
        }
    }
}

impl Flanger {
    pub fn process(&mut self, input: f32) -> f32 {
        // LFOの計算
        self.lfo_phase += self.rate / SAMPLE_RATE;
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }
        let lfo = (self.lfo_phase * TWO_PI).sin() * 0.5 + 0.5;

        // 遅延時間の計算（1ms～10ms）
        let min_delay = (0.001 * SAMPLE_RATE) as usize;
        let max_delay = (0.010 * SAMPLE_RATE) as usize;
        let delay_samples = (min_delay as f32 + (max_delay - min_delay) as f32 * self.depth * lfo) as usize;
        delay_samples.clamp(1, self.buffer.len() - 1);

        // 現在のサンプルをバッファに追加
        let delayed = self.buffer[delay_samples];
        let feedback_sample = input + delayed * self.feedback;
        self.buffer.push_back(feedback_sample);
        self.buffer.pop_front();

        // ウェットとドライの信号をミックス
        input * (1.0 - self.wet_mix) + delayed * self.wet_mix
    }
}

#[derive(Debug)]
struct Track {
    synth_type: SynthType,
    volume: f32,
    mute: bool,
    solo: bool,
    waveform: Waveform,
    frequency: f32,
    phase: f32,
    filter: Filter,
    delay: Delay,
    reverb: Reverb,
    flanger: Flanger,
    lfo: LFO,
    envelope: Envelope,
    arpeggiator: Arpeggiator,
    filter_enabled: bool,
    delay_enabled: bool,
    reverb_enabled: bool,
    flanger_enabled: bool,
    lfo_enabled: bool,
    envelope_enabled: bool,
}

impl Default for Track {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl Track {
    fn new(volume: f32) -> Self {
        Self {
            synth_type: SynthType::HarmonicDrone,
            volume,
            mute: false,
            solo: false,
            waveform: Waveform::Sine,
            frequency: 440.0,
            phase: 0.0,
            filter: Filter::default(),
            delay: Delay::default(),
            reverb: Reverb::default(),
            flanger: Flanger::default(),
            lfo: LFO::default(),
            envelope: Envelope::default(),
            arpeggiator: Arpeggiator::default(),
            filter_enabled: false,
            delay_enabled: false,
            reverb_enabled: false,
            flanger_enabled: true,
            lfo_enabled: false,
            envelope_enabled: true,
        }
    }

    fn next_sample(&mut self) -> f32 {
        // アルペジエーターの処理を追加
        if let Some(note) = self.arpeggiator.next_note() {
            self.frequency = MultiTrackSynthesizer::note_to_freq(note);
            if self.envelope_enabled {
                self.envelope.trigger();
            }
        }

        // エンベロープの値を取得
        let envelope_value = if self.envelope_enabled {
            self.envelope.next_value()
        } else {
            1.0
        };

        // エンベロープがAmplitudeで、Idleの場合は音を出さない
        if self.envelope_enabled && 
           self.envelope.target == EnvelopeTarget::Amplitude && 
           self.envelope.state == EnvelopeState::Idle {
            return 0.0;
        }

        // LFOの処理
        let lfo_value = if self.lfo_enabled {
            self.lfo.next_value()
        } else {
            0.0
        };

        // 基本波形の生成（LFOによる周波数変調を適用）
        let mut sample = self.waveform.generate_sample(self.phase);
        let freq_mod = if self.lfo_enabled && self.lfo.target == LFOTarget::Frequency {
            self.frequency * (1.0 + lfo_value)
        } else {
            self.frequency
        };
        self.phase += freq_mod / SAMPLE_RATE;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // エフェクトチェーンの適用
        if self.filter_enabled {
            sample = self.filter.process(sample);
        }
        if self.flanger_enabled {
            sample = self.flanger.process(sample);
        }
        if self.delay_enabled {
            sample = self.delay.process(sample);
        }
        if self.reverb_enabled {
            sample = self.reverb.process(sample);
        }

        // 音量の変調（エンベロープとLFO）
        let volume_mod = if self.lfo_enabled && self.lfo.target == LFOTarget::Volume {
            (1.0 + lfo_value).max(0.0)
        } else {
            1.0
        };

        let amplitude = if self.envelope_enabled && self.envelope.target == EnvelopeTarget::Amplitude {
            envelope_value
        } else {
            1.0
        };

        sample * self.volume * amplitude * volume_mod
    }
}

#[derive(Clone, Debug)]
struct TrackPreset {
    waveform: Waveform,
    frequency: f32,
    volume: f32,
    filter_enabled: bool,
    filter_mode: FilterMode,
    filter_cutoff: f32,
    filter_resonance: f32,
    delay_enabled: bool,
    delay_time: f32,
    delay_feedback: f32,
    reverb_enabled: bool,
    reverb_room_size: f32,
    reverb_wet_mix: f32,
    lfo_enabled: bool,
    lfo_frequency: f32,
    lfo_amount: f32,
    lfo_target: LFOTarget,
    envelope_enabled: bool,
    envelope_attack: f32,
    envelope_decay: f32,
    envelope_sustain: f32,
    envelope_release: f32,
}

impl TrackPreset {
    fn from_track(track: &Track) -> Self {
        Self {
            waveform: track.waveform,
            frequency: track.frequency,
            volume: track.volume,
            filter_enabled: track.filter_enabled,
            filter_mode: track.filter.mode,
            filter_cutoff: track.filter.cutoff,
            filter_resonance: track.filter.resonance,
            delay_enabled: track.delay_enabled,
            delay_time: track.delay.delay_time,
            delay_feedback: track.delay.feedback,
            reverb_enabled: track.reverb_enabled,
            reverb_room_size: track.reverb.room_size,
            reverb_wet_mix: track.reverb.wet_mix,
            lfo_enabled: track.lfo_enabled,
            lfo_frequency: track.lfo.frequency,
            lfo_amount: track.lfo.amount,
            lfo_target: track.lfo.target,
            envelope_enabled: track.envelope_enabled,
            envelope_attack: track.envelope.attack_time,
            envelope_decay: track.envelope.decay_time,
            envelope_sustain: track.envelope.sustain_level,
            envelope_release: track.envelope.release_time,
        }
    }

    fn apply_to_track(&self, track: &mut Track) {
        track.waveform = self.waveform;
        track.frequency = self.frequency;
        track.volume = self.volume;
        track.filter_enabled = self.filter_enabled;
        track.filter.mode = self.filter_mode;
        track.filter.cutoff = self.filter_cutoff;
        track.filter.resonance = self.filter_resonance;
        track.delay_enabled = self.delay_enabled;
        track.delay.delay_time = self.delay_time;
        track.delay.feedback = self.delay_feedback;
        track.reverb_enabled = self.reverb_enabled;
        track.reverb.room_size = self.reverb_room_size;
        track.reverb.wet_mix = self.reverb_wet_mix;
        track.lfo_enabled = self.lfo_enabled;
        track.lfo.frequency = self.lfo_frequency;
        track.lfo.amount = self.lfo_amount;
        track.lfo.target = self.lfo_target;
        track.envelope_enabled = self.envelope_enabled;
        track.envelope.attack_time = self.envelope_attack;
        track.envelope.decay_time = self.envelope_decay;
        track.envelope.sustain_level = self.envelope_sustain;
        track.envelope.release_time = self.envelope_release;
    }
}

struct MultiTrackSynthesizer {
    tracks: Vec<Arc<Mutex<Track>>>,
    output_stream: Option<cpal::Stream>,
    master_volume: Arc<Mutex<f32>>,
    ui_master_volume: f32,
    is_playing: Arc<Mutex<bool>>,
    selected_track: usize,
    presets: Vec<Option<TrackPreset>>,
    base_octave: i32,
}

impl Default for MultiTrackSynthesizer {
    fn default() -> Self {
        let mut synth = MultiTrackSynthesizer {
            tracks: Vec::new(),
            output_stream: None,
            master_volume: Arc::new(Mutex::new(0.2)),
            ui_master_volume: 0.2,
            is_playing: Arc::new(Mutex::new(false)),
            selected_track: 0,
            presets: vec![None, None, None],
            base_octave: 1,
        };

        // 2トラックの初期化（Channel1とChannel2で異なる設定）
        let track1 = Track::new(0.3);  // Channel1: volume 0.3
        let track2 = Track::new(0.0);  // Channel2: volume 0.0
        synth.tracks.push(Arc::new(Mutex::new(track1)));
        synth.tracks.push(Arc::new(Mutex::new(track2)));

        // オーディオ出力の設定
        if let Err(err) = synth.setup_audio() {
            eprintln!("Audio initialization error: {}", err);
        }

        synth
    }
}

impl eframe::App for MultiTrackSynthesizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ダークテーマの設定
        let mut style = (*ctx.style()).clone();
        style.visuals.dark_mode = true;
        style.visuals.window_fill = egui::Color32::from_rgb(30, 30, 30);
        style.visuals.panel_fill = egui::Color32::from_rgb(40, 40, 40);
        ctx.set_style(style);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                // ヘッダー部分
                ui.horizontal(|ui| {
                    let mut button = egui::Button::new(
                        if *self.is_playing.lock().unwrap() { "⏸" } else { "▶" }
                    );
                    if ui.add_sized(egui::vec2(40.0, 40.0), button).clicked() {
                        self.toggle_playback();
                    }

                    // マスターボリューム
                    if ui.add(
                        egui::Slider::new(&mut self.ui_master_volume, 0.0..=1.0)
                            .text("MASTER")
                    ).changed() {
                        // UIのマスターボリュームの値を実際のマスターボリュームに反映
                        if let Ok(mut master_volume) = self.master_volume.lock() {
                            *master_volume = self.ui_master_volume;
                        }
                    }

                    // プリセット
                    for (i, preset) in self.presets.iter_mut().enumerate() {
                        let text = if preset.is_some() {
                            format!("P{}", i + 1)
                        } else {
                            format!("--{}", i + 1)
                        };
                        
                        let response = ui.add_sized(egui::vec2(30.0, 30.0), egui::Button::new(text));
                        if response.clicked() {
                            if let Some(preset) = preset {
                                if let Ok(mut track) = self.tracks[self.selected_track].lock() {
                                    preset.apply_to_track(&mut track);
                                }
                            } else {
                                if let Ok(track) = self.tracks[self.selected_track].lock() {
                                    *preset = Some(TrackPreset::from_track(&track));
                                }
                            }
                        }
                        response.context_menu(|ui| {
                            if ui.button("Save").clicked() {
                                if let Ok(track) = self.tracks[self.selected_track].lock() {
                                    *preset = Some(TrackPreset::from_track(&track));
                                }
                                ui.close_menu();
                            }
                            if preset.is_some() {
                                if ui.button("Load").clicked() {
                                    if let Ok(mut track) = self.tracks[self.selected_track].lock() {
                                        if let Some(preset) = preset {
                                            preset.apply_to_track(&mut track);
                                        }
                                    }
                                    ui.close_menu();
                                }
                                if ui.button("Clear").clicked() {
                                    *preset = None;
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                });

                // キーボード
                self.draw_keyboard_ui(ui);

                // タブ形式のチャンネル選択
                ui.horizontal(|ui| {
                    for i in 0..self.tracks.len() {
                        if ui.selectable_label(
                            self.selected_track == i,
                            format!("Channel {}", i + 1)
                        ).clicked() {
                            self.selected_track = i;
                        }
                    }
                });

                // 選択されたトラックのコントロール
                if let Ok(mut track) = self.tracks[self.selected_track].lock() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            // 波形選択
                            egui::ComboBox::from_id_source("waveform")
                                .selected_text(format!("{:?}", track.waveform))
                                .width(80.0)
                                .show_ui(ui, |ui| {
                                    ui.style_mut().wrap = Some(false);
                                    ui.selectable_value(&mut track.waveform, Waveform::Sine, "Sine");
                                    ui.selectable_value(&mut track.waveform, Waveform::Square, "Square");
                                    ui.selectable_value(&mut track.waveform, Waveform::Triangle, "Triangle");
                                    ui.selectable_value(&mut track.waveform, Waveform::Sawtooth, "Saw");
                                });

                            // ボリューム
                            ui.add(
                                egui::Slider::new(&mut track.volume, 0.0..=1.0)
                                    .text("VOL")
                            );

                            // Mute/Solo
                            if ui.add_sized(
                                egui::vec2(25.0, 25.0),
                                egui::SelectableLabel::new(track.mute, "M")
                            ).clicked() {
                                track.mute = !track.mute;
                            }
                            if ui.add_sized(
                                egui::vec2(25.0, 25.0),
                                egui::SelectableLabel::new(track.solo, "S")
                            ).clicked() {
                                track.solo = !track.solo;
                            }
                        });

                        // パラメータを2段に分ける
                        ui.horizontal(|ui| {
                            // 第1段: エンベロープとエフェクトチェーン前半
                            ui.vertical(|ui| {
                                // エンベロープ
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("ENV");
                                        ui.checkbox(&mut track.envelope_enabled, "");
                                    });
                                    egui::ComboBox::from_id_source("env_target")
                                        .selected_text(format!("{:?}", track.envelope.target))
                                        .width(120.0)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(&mut track.envelope.target, EnvelopeTarget::Amplitude, "Amplitude");
                                            ui.selectable_value(&mut track.envelope.target, EnvelopeTarget::FilterCutoff, "Filter Cutoff");
                                        });
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.add(
                                                egui::Slider::new(&mut track.envelope.attack_time, 0.01..=2.0)
                                                    .text("A")
                                                    .logarithmic(true)
                                            );
                                            ui.add(
                                                egui::Slider::new(&mut track.envelope.decay_time, 0.01..=2.0)
                                                    .text("D")
                                                    .logarithmic(true)
                                            );
                                        });
                                        ui.vertical(|ui| {
                                            ui.add(
                                                egui::Slider::new(&mut track.envelope.sustain_level, 0.0..=1.0)
                                                    .text("S")
                                            );
                                            ui.add(
                                                egui::Slider::new(&mut track.envelope.release_time, 0.01..=5.0)
                                                    .text("R")
                                                    .logarithmic(true)
                                            );
                                        });
                                    });
                                });

                                // フィルター
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("FILTER");
                                        ui.checkbox(&mut track.filter_enabled, "");
                                    });
                                    egui::ComboBox::from_id_source("filter_mode")
                                        .selected_text(format!("{:?}", track.filter.mode))
                                        .width(80.0)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(&mut track.filter.mode, FilterMode::LowPass, "Low");
                                            ui.selectable_value(&mut track.filter.mode, FilterMode::HighPass, "High");
                                            ui.selectable_value(&mut track.filter.mode, FilterMode::BandPass, "Band");
                                        });
                                    ui.add(
                                        egui::Slider::new(&mut track.filter.cutoff, 20.0..=20000.0)
                                            .text("FREQ")
                                            .logarithmic(true)
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.filter.resonance, 0.1..=10.0)
                                            .text("RES")
                                    );
                                });

                                // フランジャー
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("FLANGER");
                                        ui.checkbox(&mut track.flanger_enabled, "");
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut track.flanger.rate, 0.1..=5.0)
                                            .text("RATE")
                                            .logarithmic(true)
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.flanger.depth, 0.0..=1.0)
                                            .text("DEPTH")
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.flanger.feedback, 0.0..=0.95)
                                            .text("FB")
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.flanger.wet_mix, 0.0..=1.0)
                                            .text("MIX")
                                    );
                                });
                            });

                            // 第2段: エフェクトチェーン後半とモジュレーション
                            ui.vertical(|ui| {
                                // ディレイ
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("DELAY");
                                        ui.checkbox(&mut track.delay_enabled, "");
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut track.delay.delay_time, 0.0..=2.0)
                                            .text("TIME")
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.delay.feedback, 0.0..=0.95)
                                            .text("FB")
                                    );
                                });

                                // リバーブ
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("REVERB");
                                        ui.checkbox(&mut track.reverb_enabled, "");
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut track.reverb.room_size, 0.0..=1.0)
                                            .text("SIZE")
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.reverb.wet_mix, 0.0..=1.0)
                                            .text("MIX")
                                    );
                                });

                                // LFO
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("LFO");
                                        ui.checkbox(&mut track.lfo_enabled, "");
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut track.lfo.frequency, 0.01..=10.0)
                                            .text("RATE")
                                            .logarithmic(true)
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut track.lfo.amount, 0.0..=1.0)
                                            .text("AMT")
                                    );
                                });

                                // アルペジエーター
                                ui.group(|ui| {
                                    ui.set_width(200.0);
                                    ui.horizontal(|ui| {
                                        ui.label("ARP");
                                        ui.checkbox(&mut track.arpeggiator.enabled, "");
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut track.arpeggiator.rate, 0.5..=16.0)
                                            .text("RATE")
                                            .logarithmic(true)
                                    );
                                    egui::ComboBox::from_id_source("arp_pattern")
                                        .selected_text(format!("{:?}", track.arpeggiator.pattern))
                                        .width(80.0)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(&mut track.arpeggiator.pattern, ArpPattern::Up, "Up");
                                            ui.selectable_value(&mut track.arpeggiator.pattern, ArpPattern::Down, "Down");
                                            ui.selectable_value(&mut track.arpeggiator.pattern, ArpPattern::UpDown, "Up/Down");
                                            ui.selectable_value(&mut track.arpeggiator.pattern, ArpPattern::Random, "Random");
                                        });
                                });
                            });
                        });
                    });
                }
            });
        });

        ctx.request_repaint();
    }
}

impl MultiTrackSynthesizer {
    fn setup_audio(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or("No default output device found")?;
        let config = device.default_output_config()?;

        let master_volume = Arc::clone(&self.master_volume);
        let tracks = self.tracks.clone();
        let is_playing = Arc::clone(&self.is_playing);

        let output_stream = device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let volume = *master_volume.lock().unwrap();
                let is_playing = *is_playing.lock().unwrap();
                let mut track_states: Vec<_> = tracks.iter()
                    .filter_map(|t| t.lock().ok())
                    .collect();

                // ソロトラックの存在確認
                let solo_active = track_states.iter().any(|t| t.solo);

                for sample in data.chunks_mut(2) {
                    let mut mix = 0.0;
                    
                    if is_playing {
                        for track in track_states.iter_mut() {
                            if !track.mute && (!solo_active || track.solo) {
                                mix += track.next_sample();
                            }
                        }
                    }

                    // ミックスとマスターボリューム適用
                    let final_sample = (mix * volume).clamp(-1.0, 1.0);
                    
                    if sample.len() >= 2 {
                        sample[0] = final_sample;
                        sample[1] = final_sample;
                    }
                }
            },
            |err| eprintln!("Audio output error: {}", err),
            None,
        )?;

        output_stream.play()?;
        self.output_stream = Some(output_stream);
        Ok(())
    }

    fn play(&mut self) {
        if let Ok(mut is_playing) = self.is_playing.lock() {
            *is_playing = true;
        }
    }

    fn stop(&mut self) {
        if let Ok(mut is_playing) = self.is_playing.lock() {
            *is_playing = false;
        }
    }

    fn toggle_playback(&mut self) {
        if let Ok(mut is_playing) = self.is_playing.lock() {
            *is_playing = !*is_playing;
        }
    }

    fn note_to_freq(note: i32) -> f32 {
        440.0 * 2.0f32.powf((note - 69) as f32 / 12.0)
    }

    fn trigger_note(&mut self, note: i32) {
        if let Some(track) = self.tracks.get(self.selected_track) {
            if let Ok(mut track) = track.lock() {
                if track.arpeggiator.enabled {
                    // アルペジエーターが有効な場合、アルペジエーターをトリガー
                    track.arpeggiator.trigger(note);
                } else {
                    // アルペジエーターが無効な場合、通常の音を鳴らす
                    track.frequency = Self::note_to_freq(note);
                    if track.envelope_enabled {
                        track.envelope.trigger();
                    }
                }
            }
        }
    }

    fn release_note(&mut self) {
        if let Some(track) = self.tracks.get(self.selected_track) {
            if let Ok(mut track) = track.lock() {
                if !track.arpeggiator.enabled && track.envelope_enabled {
                    track.envelope.release();
                }
            }
        }
    }

    fn draw_keyboard_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // オクターブコントロール
            ui.horizontal(|ui| {
                if ui.button("◀").clicked() {
                    self.base_octave = (self.base_octave - 1).max(0);
                }
                ui.label(format!("Octave {}", self.base_octave));
                if ui.button("▶").clicked() {
                    self.base_octave = (self.base_octave + 1).min(8);
                }
            });

            let key_height = 120.0;
            let white_key_width = 40.0;
            let black_key_width = 24.0;
            let black_key_height = key_height * 0.6;

            // キーボード全体のレイアウト用の領域を確保
            let keyboard_rect = ui.available_rect_before_wrap();
            let keyboard_response = ui.allocate_rect(keyboard_rect, egui::Sense::click_and_drag());

            // White keys
            let white_keys = ["C", "D", "E", "F", "G", "A", "B"];
            let mut note = self.base_octave * 12 + 48; // 選択されたオクターブのC
            let mut white_key_x = keyboard_rect.min.x;

            // まず白鍵を描画
            for key in white_keys {
                let key_rect = egui::Rect::from_min_size(
                    egui::pos2(white_key_x, keyboard_rect.min.y),
                    egui::vec2(white_key_width, key_height)
                );

                let response = ui.allocate_rect(key_rect, egui::Sense::click_and_drag());
                if ui.is_rect_visible(key_rect) {
                    let visuals = if response.hovered() {
                        ui.style().visuals.widgets.hovered
                    } else {
                        ui.style().visuals.widgets.inactive
                    };

                    ui.painter().rect(
                        key_rect.shrink(1.0),
                        0.0,
                        egui::Color32::WHITE,
                        visuals.bg_stroke,
                    );

                    // キーの文字を中央下部に配置
                    ui.painter().text(
                        key_rect.center_bottom() - egui::vec2(0.0, 15.0),
                        egui::Align2::CENTER_CENTER,
                        key,
                        egui::FontId::proportional(14.0),
                        egui::Color32::BLACK,
                    );
                }

                if response.dragged() || response.clicked() {
                    self.trigger_note(note);
                }
                if response.drag_released() || (!response.dragged() && !response.is_pointer_button_down_on() && response.hovered()) {
                    self.release_note();
                }

                note += match key {
                    "E" | "B" => 1,
                    _ => 2,
                };
                white_key_x += white_key_width;
            }

            // Reset for black keys
            note = self.base_octave * 12 + 49; // C#
            let mut x_offset = white_key_width - (black_key_width / 2.0);
            let black_keys = [true, true, false, true, true, true, false]; // true where black keys exist

            // 黒鍵を白鍵の上に描画
            for has_black in black_keys {
                if has_black {
                    let key_rect = egui::Rect::from_min_size(
                        egui::pos2(keyboard_rect.min.x + x_offset, keyboard_rect.min.y),
                        egui::vec2(black_key_width, black_key_height)
                    );

                    let response = ui.allocate_rect(key_rect, egui::Sense::click_and_drag());
                    if ui.is_rect_visible(key_rect) {
                        let visuals = if response.hovered() {
                            ui.style().visuals.widgets.hovered
                        } else {
                            ui.style().visuals.widgets.inactive
                        };

                        ui.painter().rect(
                            key_rect.shrink(1.0),
                            0.0,
                            egui::Color32::BLACK,
                            visuals.bg_stroke,
                        );
                    }

                    if response.dragged() || response.clicked() {
                        self.trigger_note(note);
                    }
                    if response.drag_released() || (!response.dragged() && !response.is_pointer_button_down_on() && response.hovered()) {
                        self.release_note();
                    }
                }
                note += 2;
                x_offset += white_key_width;
            }

            // キーボード領域から外れた場合のノートオフ処理
            if keyboard_response.clicked_elsewhere() || (!ui.rect_contains_pointer(keyboard_rect) && ui.input(|i| i.pointer.primary_released())) {
                self.release_note();
            }
        });
    }
}

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("2-Channel Drone Synthesizer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "2-Channel Drone Synthesizer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_pixels_per_point(1.5);
            Box::new(MultiTrackSynthesizer::default())
        }),
    );
} 