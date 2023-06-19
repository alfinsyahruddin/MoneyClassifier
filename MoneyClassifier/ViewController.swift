//
//  ViewController.swift
//  CoinDetector
//
//  Created by M Alfin Syahruddin on 17/06/23.
//

import UIKit
import AVFoundation
import Vision
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var label: UILabel!
    
    private var request: VNCoreMLRequest!

    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let videoOutputQueue = DispatchQueue(label: "video-output-queue", qos: .userInitiated)

    
    override func viewDidLoad() {
        super.viewDidLoad()
                
        setupVision()
        setupCaptureSession()
                
        DispatchQueue.global(qos: .background).async {
            self.session.startRunning()
        }
    }

    
    private func setupVision() {
        guard let moneyClassifier = try? MoneyClassifier(configuration: MLModelConfiguration()) else {
            fatalError("Failed to create an image classifier model instance.")
        }
        
        guard let model = try? VNCoreMLModel(for: moneyClassifier.model) else {
            fatalError("Failed to create a `VNCoreMLModel` instance.")
        }
        
        let request = VNCoreMLRequest(
            model: model,
            completionHandler: visionRequestHandler
        )
        self.request = request
    }
    
    
    private func setupCaptureSession() {
        session.beginConfiguration()
        
        // Add the video input to the capture session
        let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        )!
  
        // Connect the camera to the capture session input
        let cameraInput = try! AVCaptureDeviceInput(device: camera)
        session.addInput(cameraInput)
        session.sessionPreset = .vga640x480

        // Create the video data output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            String(kCVPixelBufferPixelFormatTypeKey): Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
        
        // Add the video output to the capture session
        session.addOutput(videoOutput)
        
        session.commitConfiguration()
        
        // Configure the preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = self.previewView.layer.bounds
        self.previewView.layer.addSublayer(previewLayer)
    }
    
    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
        if let error = error {
            print("Vision image detection error: \(error.localizedDescription)")
            return
        }

        if request.results == nil {
            print("Vision request had no results.")
            return
        }

        guard let observations = request.results as? [VNClassificationObservation] else {
            print("VNRequest produced the wrong result type: \(type(of: request.results)).")
            return
        }
        
        DispatchQueue.main.async {
            guard let observation = observations.first, observation.confidence > 0.9 else { return }
            
            self.label.text = observation.identifier
            switch observation.identifier {
            case "50000":
                self.label.textColor = .systemBlue
            case "100000":
                self.label.textColor = .systemRed
            default:
                break
            }
        }
    }
}


extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: imageBuffer)
            try handler.perform([request])
        } catch {
            print(error.localizedDescription)
        }
    }
}

