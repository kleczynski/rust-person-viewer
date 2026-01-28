use anyhow::Result;
use chrono::Local;
use ndarray::{s, Axis, Array3, ArrayView3};
use opencv::{core, imgcodecs, imgproc, prelude::*, videoio};
use ort::{inputs, session::Session, value::TensorRef};
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::time::sleep;
use tokio::sync::Mutex;
use tokio::task;

// --- STRUKTURY DANYCH ---

#[derive(Deserialize, Clone)]
struct Config {
    global: GlobalConfig,
    cameras: Vec<CameraConfig>,
}

#[derive(Deserialize, Clone)]
struct GlobalConfig {
    model_path: String,
}

#[derive(Deserialize, Clone)]
struct CameraConfig {
    name: String,
    url: String,
    interval_secs: u64,
    threshold: f32,
}

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    prob: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Ładowanie konfiguracji
    let config_str =
        std::fs::read_to_string("config.toml").expect("Nie znaleziono pliku config.toml");
    let config: Config = toml::from_str(&config_str)?;

    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().expect("Brak PID");

    println!("--- RUST MULTI-STREAM ORKIESTRATOR (KALIBRACJA) ---");

    // 2. Ładowanie modelu AI raz dla całego systemu
    let session = Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(&config.global.model_path)?;

    let shared_session = Arc::new(Mutex::new(session));
    let mut handles = vec![];

    // 3. Uruchamianie wątków dla każdej kamery
    for cam_cfg in config.cameras {
        let session_ref = Arc::clone(&shared_session);

        let handle = tokio::spawn(async move {
            println!("📽️ Start monitoringu: {}", cam_cfg.name);
            let _ = run_camera_loop(cam_cfg, session_ref).await;
        });

        handles.push(handle);
    }

    // 4. Wątek diagnostyki ogólnej (CPU)
    tokio::spawn(async move {
        loop {
            sys.refresh_all();

            let global_cpu = sys.global_cpu_usage();
            let process_cpu = sys
                .process(pid)
                .map(|p| p.cpu_usage())
                .unwrap_or(0.0);

            println!(
                "\n[INFO SYSTEM] CPU Global: {:.1}%, Program: {:.1}%",
                global_cpu, process_cpu
            );

            sleep(Duration::from_secs(10)).await;
        }
    });

    for h in handles {
        let _ = h.await;
    }

    Ok(())
}

async fn run_camera_loop(cfg: CameraConfig, session: Arc<Mutex<Session>>) -> Result<()> {
    loop {
        let cam_name = cfg.name.clone();
        let session_ptr = Arc::clone(&session);
        let url = cfg.url.clone();
        let thresh = cfg.threshold;

        // Operacje OpenCV i ONNX wykonujemy w wątku blokującym
        let _ = task::spawn_blocking(move || -> Result<()> {
            let mut cam = videoio::VideoCapture::from_file(&url, videoio::CAP_ANY)?;
            let mut frame = core::Mat::default();
           
            let grab_start = Instant::now();
            if cam.read(&mut frame)? && !frame.empty() {
                let grab_time = grab_start.elapsed();

                // 1. Pre-processing
                let pre_start = Instant::now();
                let input = preprocess(&frame, 640)?;
                let pre_time = pre_start.elapsed();

                // 2. Inferencja AI (z blokadą Mutex)
                let ai_start = Instant::now();
               
                // Uzyskujemy dostęp do sesji. Jeśli inna kamera już jej używa,
                // ten wątek poczeka tutaj na swoją kolej.
                let mut session_guard = session_ptr.blocking_lock();
               
                let outputs = session_guard.run(inputs![
                    "images" => TensorRef::from_array_view(&input)?
                ])?;
               
                // Zwolnienie blokady następuje automatycznie po wyjściu session_guard z zakresu
                let output_array = outputs[0]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix3>()?;
               
                let detections = postprocess(output_array.to_owned(), thresh, frame.cols() as f32, frame.rows() as f32);
                let ai_time = ai_start.elapsed();

                // 3. Obsługa wyników
                if !detections.is_empty() {
                    let filename = format!("{}_{}.jpg", cam_name, Local::now().format("%H%M%S"));
                    draw_and_save(&mut frame, &detections, &filename)?;
                   
                    let max_conf = detections.iter().map(|d| d.prob).fold(0.0, f32::max);
                    println!("[{}] 🚨 {}: WYKRYTO {} osób! (Max Conf: {:.0}%)",
                             Local::now().format("%H:%M:%S"), cam_name, detections.len(), max_conf * 100.0);
                } else {
                    let _ = imgcodecs::imwrite(&format!("last_{}.jpg", cam_name), &frame, &core::Vector::new());
                }

                // Logi wydajności dla kalibracji
                println!("📊 {}: Pobranie: {:.2?}, Pre-proc: {:.2?}, AI: {:.2?}",
                         cam_name, grab_time, pre_time, ai_time);
            }
            Ok(())
        }).await?;

        // Odczekaj interwał zdefiniowany w TOML
        sleep(Duration::from_secs(cfg.interval_secs)).await;
    }
}


fn preprocess(frame: &core::Mat, size: i32) -> Result<ndarray::Array4<f32>> {
    let mut resized = core::Mat::default();
    imgproc::resize(frame, &mut resized, core::Size::new(size, size), 0., 0., imgproc::INTER_LINEAR)?;
    let mut rgb = core::Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let bytes = rgb.data_bytes()?;
    let shape = (size as usize, size as usize, 3);
    let array = ArrayView3::from_shape(shape, bytes)?.mapv(|x| x as f32 / 255.0);
    let blob = array.permuted_axes([2, 0, 1]).insert_axis(Axis(0)).as_standard_layout().to_owned();
    Ok(blob)
}

fn postprocess(output: Array3<f32>, threshold: f32, w_orig: f32, h_orig: f32) -> Vec<BoundingBox> {
    let mut raw_results = Vec::new();
    let view = output.slice(s![0, .., ..]);
    let view_t = view.t();

    for row in view_t.axis_iter(Axis(0)) {
        let prob = row[4usize];
        if prob > threshold {
            let xc = row[0usize] / 640. * w_orig;
            let yc = row[1usize] / 640. * h_orig;
            let w = row[2usize] / 640. * w_orig;
            let h = row[3usize] / 640. * h_orig;

            raw_results.push(BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
                prob,
            });
        }
    }

    raw_results.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    let mut final_results = Vec::new();
    while !raw_results.is_empty() {
        let best = raw_results.remove(0);
        final_results.push(best);
        raw_results.retain(|item| calculate_iou(&best, item) < 0.45);
    }
    final_results
}

fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x_left = box1.x1.max(box2.x1);
    let y_top = box1.y1.max(box2.y1);
    let x_right = box1.x2.min(box2.x2);
    let y_bottom = box1.y2.min(box2.y2);
    if x_right < x_left || y_bottom < y_top { return 0.0; }
    let intersection_area = (x_right - x_left) * (y_bottom - y_top);
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    intersection_area / (area1 + area2 - intersection_area)
}

// --- ZMODYFIKOWANA FUNKCJA RYSOWANIA ---
fn draw_and_save(frame: &mut core::Mat, detections: &[BoundingBox], name: &str) -> Result<()> {
    for bbox in detections {
        // 1. Rysowanie ramki
        let rect = core::Rect::new(
            bbox.x1 as i32, bbox.y1 as i32,
            (bbox.x2 - bbox.x1) as i32, (bbox.y2 - bbox.y1) as i32,
        );
        imgproc::rectangle(frame, rect, core::Scalar::new(0., 0., 255., 0.), 2, 8, 0)?;

        // 2. Przygotowanie tekstu (np. "Osoba: 85%")
        let label = format!("Osoba: {:.0}%", bbox.prob * 100.0);
       
        // Punkt, w którym zacznie się tekst (lekko nad ramką)
        let text_pos = core::Point::new(bbox.x1 as i32, (bbox.y1 as i32) - 10);

        // 3. Naniesienie tekstu na obraz
        imgproc::put_text(
            frame,
            &label,
            text_pos,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,                         // Skala czcionki
            core::Scalar::new(0., 0., 255., 0.), // Kolor czerwony
            2,                           // Grubość kreski
            imgproc::LINE_8,
            false
        )?;
    }
    imgcodecs::imwrite(name, frame, &core::Vector::new())?;
    Ok(())
}