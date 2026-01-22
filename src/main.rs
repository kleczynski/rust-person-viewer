use anyhow::{Context, Result};
use chrono::Local;
use ndarray::{s, Array4, Axis};
use opencv::{core, imgcodecs, imgproc, prelude::*, videoio};
use ort::{inputs, value::TensorRef, Session};
use std::{
    thread,
    time::{Duration, Instant},
};

// --- STRUKTURY POMOCNICZE ---
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let rtsp_url = std::env::var("RTSP_URL")?;
    let model_path = std::env::var("MODEL_PATH")?;
    let threshold: f32 = std::env::var("THRESHOLD")?.parse()?;

    // 1. Inicjalizacja modelu (raz na starcie)
    let session = Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    println!("🕵️ Strażnik uruchomiony. Analiza co 30 sekund...");

    loop {
        let start_time = Instant::now();

        // 2. Krótkie połączenie z kamerą (otwieramy, bierzemy klatkę, zamykamy)
        // To najlepsze dla RAM-u – nie trzymamy otwartego strumienia.
        if let Ok(mut cam) = videoio::VideoCapture::from_file(&rtsp_url, videoio::CAP_ANY) {
            let mut frame = core::Mat::default();

            // Przeskakujemy kilka klatek, aby uniknąć ewentualnego "szumu" przy starcie połączenia
            for _ in 0..5 {
                cam.read(&mut frame)?;
            }

            if !frame.empty() {
                // 3. Preprocessing
                let input = preprocess(&frame, 640)?;

                // 4. Inferencja
                let outputs =
                    session.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
                let output = outputs["output0"]
                    .try_extract_array::<f32>()?
                    .t()
                    .into_owned();

                // 5. Sprawdzanie czy jest osoba (indeks klasy 0 lub 4 w zależności od modelu)
                let detections =
                    postprocess(output, threshold, frame.cols() as f32, frame.rows() as f32);

                if !detections.is_empty() {
                    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
                    let filename = format!("person_{}.jpg", timestamp);

                    // Rysujemy ramki dla dowodu
                    draw_and_save(&mut frame, &detections, &filename)?;
                    println!("📸 Wykryto osobę! Zdjęcie zapisane: {}", filename);
                }
            }
        } else {
            eprintln!("❌ Błąd połączenia z kamerą. Spróbuję za 30s.");
        }

        // 6. Odpoczynek dla serwera - dokładnie 30 sekund
        let elapsed = start_time.elapsed();
        if elapsed < Duration::from_secs(30) {
            thread::sleep(Duration::from_secs(30) - elapsed);
        }
    }
}

// --- FUNKCJE POMOCNICZE ---

fn preprocess(frame: &core::Mat, size: i32) -> Result<Array4<f32>> {
    let mut resized = core::Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        core::Size::new(size, size),
        0.,
        0.,
        imgproc::INTER_LINEAR,
    )?;
    let mut rgb = core::Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    let mut f32_mat = core::Mat::default();
    rgb.convert_to(&mut f32_mat, core::CV_32F, 1.0 / 255.0, 0.0)?;

    let data: Vec<f32> = f32_mat.data_typed::<f32>()?.to_vec();
    let array = ndarray::Array::from_shape_vec((size as usize, size as usize, 3), data)?;
    Ok(array
        .permuted_axes([2, 0, 1])
        .insert_axis(Axis(0))
        .to_owned())
}

fn postprocess(
    output: ndarray::Array2<f32>,
    threshold: f32,
    w_orig: f32,
    h_orig: f32,
) -> Vec<BoundingBox> {
    let mut results = Vec::new();
    let view = output.slice(s![.., .., 0]);

    for row in view.axis_iter(Axis(0)) {
        let prob = row[4]; // Zazwyczaj klasa person
        if prob > threshold {
            let xc = row[0] / 640. * w_orig;
            let yc = row[1] / 640. * h_orig;
            let w = row[2] / 640. * w_orig;
            let h = row[3] / 640. * h_orig;
            results.push(BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            });
        }
    }
    results // W tym prostym skrypcie bierzemy surowe dane (można dodać NMS)
}

fn draw_and_save(frame: &mut core::Mat, detections: &[BoundingBox], name: &str) -> Result<()> {
    for bbox in detections {
        let rect = core::Rect::new(
            bbox.x1 as i32,
            bbox.y1 as i32,
            (bbox.x2 - bbox.x1) as i32,
            (bbox.y2 - bbox.y1) as i32,
        );
        imgproc::rectangle(frame, rect, core::Scalar::new(0., 0., 255., 0.), 2, 8, 0)?;
    }
    imgcodecs::imwrite(name, frame, &core::Vector::new())?;
    Ok(())
}
