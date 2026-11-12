// TTS Lab — Qt 6 front end for libttslab.
//
// All synthesis runs on a worker thread (QtConcurrent) so the first use of
// a data-driven engine (a couple of seconds of voice building) doesn't
// freeze the window. Only one job runs at a time: the library handle is not
// thread-safe, so the controls are disabled while a job is in flight.

#include <QtWidgets>
#include <QAudioFormat>
#include <QAudioSink>
#include <QMediaDevices>
#include <QtConcurrent>
#include <cmath>
#include <deque>
#include <vector>

#include "ttslab.h"

#ifndef TTSLAB_ROOT
#define TTSLAB_ROOT "."
#endif

struct Rendered {
    bool ok = false;
    std::vector<float> samples;
    std::vector<float> f0;
    float f0Rate = 200;
    QString transcription;
    double audioSeconds = 0, synthMs = 0, buildMs = 0;
    int units = 0, joins = 0, fallbacks = 0;
    QImage spectrogram;
};

static QImage makeSpectrogram(const std::vector<float> &s) {
    const int hop = 80;
    const size_t maxFrames = s.size() / hop + 1;
    std::vector<float> db(maxFrames * 256);
    const size_t frames = ttslab_spectrogram(s.data(), s.size(), hop, db.data(), maxFrames);
    float top = -200;
    for (size_t i = 0; i < frames * 256; ++i) top = std::max(top, db[i]);
    QImage img(int(frames), 256, QImage::Format_RGB32);
    for (size_t f = 0; f < frames; ++f) {
        for (int k = 0; k < 256; ++k) {
            // 70 dB of dynamic range, low frequencies at the bottom.
            float v = std::clamp((db[f * 256 + k] - (top - 70)) / 70.0f, 0.0f, 1.0f);
            const int r = int(255 * std::pow(v, 0.7f));
            const int g = int(255 * std::pow(v, 1.8f));
            const int b = int(255 * std::clamp(0.5f - std::abs(v - 0.35f) * 1.5f, 0.0f, 0.5f) * 2);
            img.setPixel(int(f), 255 - k, qRgb(r, g, b));
        }
    }
    return img;
}

class PlotWidget : public QWidget {
public:
    explicit PlotWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setMinimumHeight(260);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    void setData(const Rendered &r) {
        samples_ = r.samples;
        f0_ = r.f0;
        f0Rate_ = r.f0Rate;
        spec_ = r.spectrogram;
        update();
    }

protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.fillRect(rect(), QColor(20, 20, 24));
        if (samples_.empty()) {
            p.setPen(Qt::gray);
            p.drawText(rect(), Qt::AlignCenter, "Type some text and press Speak");
            return;
        }
        const int W = width();
        const int specH = int(height() * 0.72);
        const QRect specRect(0, 0, W, specH);
        p.drawImage(specRect, spec_);

        // Planned F0 contour over the spectrogram (log scale 50..400 Hz on
        // the right-hand axis), so you can see prosody against the audio.
        const double dur = samples_.size() / 16000.0;
        auto yOf = [&](float hz) {
            const double t = (std::log(hz) - std::log(50.0)) / (std::log(400.0) - std::log(50.0));
            return specH - t * specH;
        };
        p.setPen(QPen(QColor(80, 220, 255), 2));
        QPainterPath path;
        for (size_t i = 0; i < f0_.size(); ++i) {
            const double x = (i / f0Rate_) / dur * W;
            if (x > W) break;
            const QPointF pt(x, yOf(std::max(50.0f, f0_[i])));
            if (i == 0) path.moveTo(pt); else path.lineTo(pt);
        }
        p.drawPath(path);
        p.setPen(QColor(80, 220, 255));
        for (int hz : {100, 200, 300})
            p.drawText(W - 52, int(yOf(float(hz))) - 2, QString("%1 Hz").arg(hz));
        p.setPen(QColor(230, 230, 230));
        p.drawText(4, 14, "0-8 kHz spectrogram, planned F0 in blue");

        // Waveform envelope.
        const int wy = specH + 4, wh = height() - specH - 8;
        p.fillRect(0, wy, W, wh, QColor(30, 30, 36));
        p.setPen(QColor(150, 230, 150));
        const double per = double(samples_.size()) / W;
        for (int x = 0; x < W; ++x) {
            const size_t a = size_t(x * per), b = std::min(samples_.size(), size_t((x + 1) * per) + 1);
            float lo = 0, hi = 0;
            for (size_t i = a; i < b; ++i) { lo = std::min(lo, samples_[i]); hi = std::max(hi, samples_[i]); }
            p.drawLine(x, int(wy + wh / 2 - hi * wh / 2), x, int(wy + wh / 2 - lo * wh / 2));
        }
    }

private:
    std::vector<float> samples_, f0_;
    float f0Rate_ = 200;
    QImage spec_;
};

static QByteArray toWav(const std::vector<float> &s, int rate) {
    QByteArray out;
    QDataStream ds(&out, QIODevice::WriteOnly);
    ds.setByteOrder(QDataStream::LittleEndian);
    const quint32 dataLen = quint32(s.size() * 2);
    ds.writeRawData("RIFF", 4); ds << quint32(36 + dataLen);
    ds.writeRawData("WAVEfmt ", 8); ds << quint32(16) << quint16(1) << quint16(1)
                                       << quint32(rate) << quint32(rate * 2) << quint16(2) << quint16(16);
    ds.writeRawData("data", 4); ds << dataLen;
    for (float v : s) ds << qint16(std::clamp(v, -1.0f, 1.0f) * 32767);
    return out;
}

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        tts_ = ttslab_create();
        ttslab_default_options(&opt_);
        loadDictionary();
        buildUi();
        setWindowTitle("TTS Lab");
        resize(1100, 760);
    }
    ~MainWindow() override {
        watcher_.waitForFinished();
        ttslab_destroy(tts_);
    }

    void autoScreenshot(const QString &path) {
        autoplay_->setChecked(false);
        speak();
        connect(&watcher_, &QFutureWatcher<Rendered>::finished, this, [this, path] {
            QTimer::singleShot(300, this, [this, path] {
                grab().save(path);
                qApp->quit();
            });
        });
    }

private:
    void loadDictionary() {
        // CMUdict if tools/fetch-cmudict.sh has been run; the built-in
        // ~350-word list plus letter-to-sound rules otherwise.
        for (const QString &p : {QString("data/cmudict.dict"), QString(TTSLAB_ROOT "/data/cmudict.dict")}) {
            QFile f(p);
            if (f.open(QIODevice::ReadOnly)) {
                const QByteArray d = f.readAll();
                ttslab_add_dict(tts_, d.constData(), size_t(d.size()));
                return;
            }
        }
    }

    void buildUi() {
        auto *central = new QWidget;
        auto *root = new QHBoxLayout(central);

        // ---- left column: controls
        auto *controls = new QVBoxLayout;
        auto *engineBox = new QGroupBox("Engine");
        auto *ev = new QVBoxLayout(engineBox);
        engine_ = new QComboBox;
        for (int i = 0; i < ttslab_engine_count(); ++i) engine_->addItem(ttslab_engine_name(i));
        engine_->setCurrentIndex(opt_.engine);
        ev->addWidget(engine_);
        psola_ = new QCheckBox("Diphone: TD-PSOLA prosody");
        modify_ = new QCheckBox("Unit selection: force planned prosody");
        mlpg_ = new QCheckBox("HMM: MLPG trajectory smoothing");
        gv_ = new QCheckBox("HMM: global-variance postfilter");
        psola_->setChecked(opt_.diphone_psola);
        modify_->setChecked(opt_.unit_modify);
        mlpg_->setChecked(opt_.hmm_mlpg);
        gv_->setChecked(opt_.hmm_gv);
        for (auto *c : {psola_, modify_, mlpg_, gv_}) ev->addWidget(c);
        engineHelp_ = new QLabel;
        engineHelp_->setWordWrap(true);
        engineHelp_->setStyleSheet("color: gray; font-size: 11px;");
        ev->addWidget(engineHelp_);
        controls->addWidget(engineBox);

        auto *prosBox = new QGroupBox("Prosody and voice");
        auto *form = new QFormLayout(prosBox);
        f0model_ = new QComboBox;
        f0model_->addItems({"Flat (monotone)", "Hat pattern (rules)", "Fujisaki model"});
        f0model_->setCurrentIndex(opt_.f0_model);
        form->addRow("F0 model", f0model_);
        auto spin = [&](double v, double lo, double hi) {
            auto *s = new QDoubleSpinBox;
            s->setRange(lo, hi); s->setSingleStep(0.05); s->setValue(v);
            return s;
        };
        pitch_ = spin(opt_.pitch_scale, 0.3, 3.0);
        rate_ = spin(opt_.rate, 0.3, 4.0);
        range_ = spin(opt_.pitch_range, 0.0, 3.0);
        formant_ = spin(opt_.formant_scale, 0.6, 1.6);
        form->addRow("Pitch ×", pitch_);
        form->addRow("Rate ×", rate_);
        form->addRow("Pitch range ×", range_);
        form->addRow("Vocal tract ×", formant_);
        auto *presets = new QHBoxLayout;
        for (auto [name, p, f] : {std::tuple{"Man", 1.0, 1.0}, {"Woman", 1.8, 1.17}, {"Child", 2.3, 1.3}}) {
            auto *b = new QPushButton(name);
            connect(b, &QPushButton::clicked, this, [=, this] { pitch_->setValue(p); formant_->setValue(f); });
            presets->addWidget(b);
        }
        form->addRow("Presets", presets);
        controls->addWidget(prosBox);

        autoplay_ = new QCheckBox("Play after synthesis");
        autoplay_->setChecked(true);
        controls->addWidget(autoplay_);
        controls->addStretch();
        auto *benchBtn = new QPushButton("Run cost profile…");
        controls->addWidget(benchBtn);
        root->addLayout(controls, 0);

        // ---- right column: text, buttons, output
        auto *right = new QVBoxLayout;
        text_ = new QPlainTextEdit("Hello! This is a toy speech synthesizer. Can you hear the difference between the engines?");
        text_->setMaximumHeight(90);
        right->addWidget(text_);
        auto *buttons = new QHBoxLayout;
        speakBtn_ = new QPushButton("Speak");
        speakBtn_->setDefault(true);
        auto *allBtn = new QPushButton("Speak with every engine");
        auto *playBtn = new QPushButton("Replay");
        auto *saveBtn = new QPushButton("Save WAV…");
        for (auto *b : {speakBtn_, allBtn, playBtn, saveBtn}) buttons->addWidget(b);
        right->addLayout(buttons);
        transcription_ = new QLabel;
        transcription_->setWordWrap(true);
        transcription_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        transcription_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        right->addWidget(transcription_);
        stats_ = new QLabel;
        right->addWidget(stats_);
        plot_ = new PlotWidget;
        right->addWidget(plot_, 1);
        root->addLayout(right, 1);
        setCentralWidget(central);
        statusBar()->showMessage(QString("Lexicon: %1 words").arg(ttslab_dict_words(tts_)));

        connect(engine_, &QComboBox::currentIndexChanged, this, [this] { updateEnabled(); });
        connect(speakBtn_, &QPushButton::clicked, this, [this] { speak(); });
        connect(allBtn, &QPushButton::clicked, this, [this] { speakAll(); });
        connect(playBtn, &QPushButton::clicked, this, [this] { play(); });
        connect(saveBtn, &QPushButton::clicked, this, [this] { save(); });
        connect(benchBtn, &QPushButton::clicked, this, [this] { bench(); });
        updateEnabled();
    }

    void updateEnabled() {
        const int e = engine_->currentIndex();
        psola_->setEnabled(e == TTSLAB_DIPHONE);
        modify_->setEnabled(e == TTSLAB_UNIT_SELECTION);
        mlpg_->setEnabled(e == TTSLAB_HMM);
        gv_->setEnabled(e == TTSLAB_HMM);
        formant_->setEnabled(e == TTSLAB_FORMANT_CLASSIC || e == TTSLAB_FORMANT_REFINED || e == TTSLAB_HMM);
        static const char *help[] = {
            "Rules → impulse source → resonators. Straight-line formant transitions, 5 ms parameter hold. ~1980.",
            "Same synthesizer with an LF glottal model, coarticulation (dominance functions), jitter, breathiness, microprosody.",
            "One recorded unit per phone pair, joined and re-pitched with TD-PSOLA. Untick PSOLA to hear raw units.",
            "Viterbi search over every diphone in a sentence corpus. Prefers long contiguous stretches of recording.",
            "Gaussian state models over cepstra, MLPG trajectories, pulse+noise vocoder. Smooth but muffled.",
        };
        engineHelp_->setText(help[e]);
        if (!ttslab_engine_ready(tts_, e)) engineHelp_->setText(engineHelp_->text() + "\n(voice is built on first use, a few seconds)");
    }

    ttslab_options gather() const {
        ttslab_options o = opt_;
        o.engine = engine_->currentIndex();
        o.f0_model = f0model_->currentIndex();
        o.pitch_scale = float(pitch_->value());
        o.rate = float(rate_->value());
        o.pitch_range = float(range_->value());
        o.formant_scale = float(formant_->value());
        o.diphone_psola = psola_->isChecked();
        o.unit_modify = modify_->isChecked();
        o.hmm_mlpg = mlpg_->isChecked();
        o.hmm_gv = gv_->isChecked();
        return o;
    }

    static Rendered render(ttslab *t, const QByteArray &text, ttslab_options o) {
        Rendered out;
        ttslab_result r;
        if (ttslab_speak(t, text.constData(), &o, &r) != 0) return out;
        out.ok = true;
        out.samples.assign(r.samples, r.samples + r.n_samples);
        out.f0.assign(r.f0, r.f0 + r.n_f0);
        out.f0Rate = r.f0_frame_rate;
        out.transcription = QString::fromUtf8(r.transcription);
        out.audioSeconds = r.audio_seconds;
        out.synthMs = r.synth_ms;
        out.buildMs = r.build_ms;
        out.units = r.units; out.joins = r.joins; out.fallbacks = r.fallbacks;
        ttslab_result_free(&r);
        out.spectrogram = makeSpectrogram(out.samples);
        return out;
    }

    void setBusy(bool b, const QString &msg = {}) {
        centralWidget()->setEnabled(!b);
        statusBar()->showMessage(msg);
    }

    void speak() {
        if (watcher_.isRunning()) return;
        const ttslab_options o = gather();
        const QByteArray text = text_->toPlainText().toUtf8();
        setBusy(true, ttslab_engine_ready(tts_, o.engine) ? "Synthesizing…" : "Building voice (first use)…");
        auto *t = tts_;
        watcher_.disconnect();
        connect(&watcher_, &QFutureWatcher<Rendered>::finished, this, [this, o] {
            setBusy(false);
            showResult(watcher_.result(), o);
            updateEnabled();
        });
        watcher_.setFuture(QtConcurrent::run([t, text, o] { return render(t, text, o); }));
    }

    void speakAll() {
        if (watcher_.isRunning()) return;
        // Render each engine in turn, playing each result; chained through
        // the watcher so the UI stays responsive.
        queue_.clear();
        for (int e = 0; e < ttslab_engine_count(); ++e) queue_.push_back(e);
        nextInQueue();
    }

    void nextInQueue() {
        if (queue_.empty()) return;
        const int e = queue_.front();
        queue_.pop_front();
        engine_->setCurrentIndex(e);
        ttslab_options o = gather();
        const QByteArray text = text_->toPlainText().toUtf8();
        setBusy(true, QString("Engine %1/%2…").arg(e + 1).arg(ttslab_engine_count()));
        auto *t = tts_;
        watcher_.disconnect();
        connect(&watcher_, &QFutureWatcher<Rendered>::finished, this, [this, o] {
            setBusy(false);
            showResult(watcher_.result(), o);
            // Wait for playback to end before the next engine.
            const int ms = int(last_.audioSeconds * 1000) + 300;
            QTimer::singleShot(autoplay_->isChecked() ? ms : 0, this, [this] { nextInQueue(); });
        });
        watcher_.setFuture(QtConcurrent::run([t, text, o] { return render(t, text, o); }));
    }

    void showResult(const Rendered &r, const ttslab_options &o) {
        if (!r.ok) {
            QMessageBox::warning(this, "TTS Lab", "Synthesis failed.");
            return;
        }
        last_ = r;
        transcription_->setText(r.transcription);
        QString s = QString("%1 — %2 s of audio in %3 ms (%4× real time)")
                        .arg(ttslab_engine_name(o.engine))
                        .arg(r.audioSeconds, 0, 'f', 2)
                        .arg(r.synthMs, 0, 'f', 1)
                        .arg(r.audioSeconds * 1000 / std::max(0.01, r.synthMs), 0, 'f', 0);
        if (r.buildMs > 10) s += QString(" · voice built in %1 s").arg(r.buildMs / 1000, 0, 'f', 2);
        if (o.engine == TTSLAB_UNIT_SELECTION)
            s += QString(" · %1 units, %2 joins, %3 fallbacks").arg(r.units).arg(r.joins).arg(r.fallbacks);
        stats_->setText(s);
        plot_->setData(r);
        if (autoplay_->isChecked()) play();
    }

    void play() {
        if (last_.samples.empty()) return;
        if (sink_) { sink_->stop(); delete sink_; sink_ = nullptr; }
        QAudioFormat fmt;
        fmt.setSampleRate(16000);
        fmt.setChannelCount(1);
        fmt.setSampleFormat(QAudioFormat::Int16);
        pcm_.resize(qsizetype(last_.samples.size() * 2));
        auto *d = reinterpret_cast<qint16 *>(pcm_.data());
        for (size_t i = 0; i < last_.samples.size(); ++i) d[i] = qint16(std::clamp(last_.samples[i], -1.0f, 1.0f) * 32767);
        buffer_.close();
        buffer_.setData(pcm_);
        buffer_.open(QIODevice::ReadOnly);
        sink_ = new QAudioSink(QMediaDevices::defaultAudioOutput(), fmt, this);
        sink_->start(&buffer_);
    }

    void save() {
        if (last_.samples.empty()) return;
        const QString path = QFileDialog::getSaveFileName(this, "Save WAV", "ttslab.wav", "WAV (*.wav)");
        if (path.isEmpty()) return;
        QFile f(path);
        if (f.open(QIODevice::WriteOnly)) f.write(toWav(last_.samples, 16000));
    }

    void bench() {
        if (watcher_.isRunning()) return;
        setBusy(true, "Building every voice and timing each engine…");
        const ttslab_options o = gather();
        auto *t = tts_;
        auto *bw = new QFutureWatcher<QString>(this);
        connect(bw, &QFutureWatcher<QString>::finished, this, [this, bw] {
            setBusy(false);
            auto *dlg = new QDialog(this);
            dlg->setWindowTitle("Cost profile");
            auto *lay = new QVBoxLayout(dlg);
            auto *view = new QTextBrowser;
            view->setMarkdown(bw->result());
            lay->addWidget(view);
            dlg->resize(900, 360);
            dlg->show();
            bw->deleteLater();
        });
        bw->setFuture(QtConcurrent::run([t, o] {
            char *md = ttslab_bench(t, &o, 5);
            QString s = md ? QString::fromUtf8(md) : QString("Benchmark failed.");
            ttslab_string_free(md);
            return s;
        }));
    }

    ttslab *tts_ = nullptr;
    ttslab_options opt_{};
    QComboBox *engine_, *f0model_;
    QCheckBox *psola_, *modify_, *mlpg_, *gv_, *autoplay_;
    QDoubleSpinBox *pitch_, *rate_, *range_, *formant_;
    QPlainTextEdit *text_;
    QPushButton *speakBtn_;
    QLabel *engineHelp_, *transcription_, *stats_;
    PlotWidget *plot_;
    QFutureWatcher<Rendered> watcher_;
    std::deque<int> queue_;
    Rendered last_;
    QAudioSink *sink_ = nullptr;
    QByteArray pcm_;
    QBuffer buffer_;
};

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    // TTSLAB_SCREENSHOT=file.png: speak once, save a screenshot, quit
    // (for docs and headless checks: QT_QPA_PLATFORM=offscreen).
    if (const char *shot = std::getenv("TTSLAB_SCREENSHOT")) {
        w.autoScreenshot(QString::fromLocal8Bit(shot));
    }
    return app.exec();
}
