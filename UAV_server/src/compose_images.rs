use opencv::{
    core::{Mat, Point2f, Scalar},
    imgproc,
    calib3d,
    prelude::*,
    types,
};
use anyhow::Result;
use rand::Rng;
use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;

fn compose_images(rgb_image: &Mat, ir_image: &Mat) -> Result<Mat> {
    let keypoints_rgb = detect_keypoints(rgb_image)?;
    let keypoints_ir = detect_keypoints(ir_image)?;

    let descriptors_rgb = compute_descriptors(rgb_image, &keypoints_rgb)?;
    let descriptors_ir = compute_descriptors(ir_image, &keypoints_ir)?;

    let matches = match_keypoints(&descriptors_rgb, &descriptors_ir)?;
    let homography = compute_homography(&keypoints_rgb, &keypoints_ir, &matches)?;

    let mut ir_transformed = Mat::default();
    imgproc::warp_perspective(
        ir_image,
        &mut ir_transformed,
        &homography,
        rgb_image.size()?,
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    let enhanced_ir = enhance_contrast_via_cpp(&ir_transformed)?;

    let composed_image = weighted_image_fusion(rgb_image, &enhanced_ir, 0.6, 0.4, 10.0)?;

    Ok(composed_image)
}

fn detect_keypoints(image: &Mat) -> Result<types::VectorOfKeyPoint> {
    let detector = <dyn opencv::features2d::ORB>::create(500, 1.2, 8, 31, 0, 2, opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
    let mut keypoints = types::VectorOfKeyPoint::new();
    detector.detect(image, &mut keypoints, &opencv::core::no_array())?;
    Ok(keypoints)
}

fn compute_descriptors(image: &Mat, keypoints: &types::VectorOfKeyPoint) -> Result<Mat> {
    let extractor = <dyn opencv::features2d::ORB>::create(500, 1.2, 8, 31, 0, 2, opencv::features2d::ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
    let mut descriptors = Mat::default();
    extractor.compute(image, keypoints, &mut descriptors)?;
    Ok(descriptors)
}

fn match_keypoints(descriptors_rgb: &Mat, descriptors_ir: &Mat) -> Result<types::VectorOfDMatch> {
    let matcher = <dyn opencv::features2d::BFMatcher>::new(opencv::core::NORM_HAMMING, true)?;
    let mut matches = types::VectorOfDMatch::new();
    matcher.match_(descriptors_rgb, descriptors_ir, &mut matches, &opencv::core::no_array())?;
    matches.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    Ok(matches)
}

fn compute_homography(
    keypoints_rgb: &types::VectorOfKeyPoint,
    keypoints_ir: &types::VectorOfKeyPoint,
    matches: &types::VectorOfDMatch,
) -> Result<Mat> {
    let points_rgb: Vec<Point2f> = matches.iter().map(|m| keypoints_rgb.get(m.query_idx as usize).unwrap().pt).collect();
    let points_ir: Vec<Point2f> = matches.iter().map(|m| keypoints_ir.get(m.train_idx as usize).unwrap().pt).collect();
    let homography = calib3d::find_homography(&points_ir, &points_rgb, calib3d::RANSAC, 3.0, &mut Mat::default(), 2000, 0.995)?;
    Ok(homography)
}

fn weighted_image_fusion(rgb: &Mat, ir: &Mat, alpha: f64, beta: f64, gamma: f64) -> Result<Mat> {
    let mut fused_image = Mat::default();
    opencv::core::add_weighted(rgb, alpha, ir, beta, gamma, &mut fused_image, -1)?;
    Ok(fused_image)
}

extern "C" {
    fn enhance_image_contrast(image_data: *const u8, width: i32, height: i32, channels: i32) -> *mut u8;
}

fn enhance_contrast_via_cpp(image: &Mat) -> Result<Mat> {
    let (width, height, channels) = (image.cols(), image.rows(), image.channels());
    let image_data = image.data_bytes()?;
    let enhanced_ptr = unsafe { enhance_image_contrast(image_data.as_ptr(), width, height, channels) };
    let enhanced_slice = unsafe { slice::from_raw_parts(enhanced_ptr, (width * height * channels) as usize) };
    Mat::from_slice(enhanced_slice).map_err(Into::into)
}

fn calculate_contrast(image: &Mat) -> Result<f64> {
    let mean = opencv::core::mean(&image, &opencv::core::no_array())?;
    let stddev = opencv::core::mean_std_dev(&image)?.1;
    Ok(stddev.iter().sum::<f64>() / mean.iter().sum::<f64>())
}

fn adaptive_weighting(rgb_contrast: f64, ir_contrast: f64) -> (f64, f64) {
    let total = rgb_contrast + ir_contrast;
    (rgb_contrast / total, ir_contrast / total)
}

fn optimized_weighted_fusion(rgb: &Mat, ir: &Mat) -> Result<Mat> {
    let rgb_contrast = calculate_contrast(rgb)?;
    let ir_contrast = calculate_contrast(ir)?;
    let (alpha, beta) = adaptive_weighting(rgb_contrast, ir_contrast);
    weighted_image_fusion(rgb, ir, alpha, beta, 0.0)
}
