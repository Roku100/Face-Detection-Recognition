# Development Notes

## 2026-02-06

### Face Encoding Approach

Initially tried using the face_recognition library but ran into Python 3.13 compatibility issues with dlib. Switched to a custom implementation using Local Binary Patterns (LBP) for feature extraction.

LBP works pretty well for this use case - it's lighting-invariant and faster than deep learning approaches. The chi-square distance metric seems to work better than Euclidean for comparing histograms.

### Flickering Issue

Had an issue where the name label was flickering between recognized/unknown. Fixed by adding a face tracker that maintains identity across frames using IoU matching and confidence voting over the last 5 frames.

### Performance

- Haar Cascade: ~30 FPS, decent accuracy
- DNN (ResNet-SSD): ~15 FPS, better accuracy
- Frame skip of 2 seems like a good balance

### TODO

- [ ] Add support for multiple cameras
- [ ] Implement face alignment before encoding
- [ ] Add confidence threshold adjustment in UI
- [ ] Test with more diverse lighting conditions
- [ ] Maybe add MTCNN detector option

### Known Issues

- LBP encoding is slower than expected due to nested loops - could vectorize this
- Database grows large with many people - might need to add compression
- Sometimes fails to detect faces at extreme angles

## Configuration

Current tolerance of 0.6 works well for most cases. Lower values (0.4-0.5) reduce false positives but might miss some valid matches.

Quality threshold of 0.7 for registration ensures good samples are captured.
