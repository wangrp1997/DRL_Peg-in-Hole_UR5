/*
 * Copied from third_party/visp/example/direct-visual-servoing/photometricVisualServoing.cpp
 * (Collewet08c). Only the vpFeatureLuminance + vpServo task is kept; image acquisition
 * is external (PyBullet wrist_camera2).
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpImage.h>
#include <visp3/visual_features/vpFeatureLuminance.h>
#include <visp3/vs/vpServo.h>

namespace py = pybind11;

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

class PhotometricServoTask
{
public:
  void init(vpImage<unsigned char> &Id, const vpCameraParameters &cam, double Z, double lambda)
  {
    // photometricVisualServoing.cpp lines 346-370
    sI.init(Id.getHeight(), Id.getWidth(), Z);
    sId.init(Id.getHeight(), Id.getWidth(), Z);
    sI.setCameraParameters(cam);
    sId.setCameraParameters(cam);
    sI.buildFrom(Id);
    sId.buildFrom(Id);

    servo.setServo(vpServo::EYEINHAND_CAMERA);
    servo.addFeature(sI, sId);
    servo.setLambda(lambda);
    servo.setInteractionMatrixType(vpServo::CURRENT);
    initialized = true;
  }

  py::tuple step(vpImage<unsigned char> &I)
  {
    if (!initialized) {
      throw std::runtime_error("PhotometricServoTask::init() must be called first");
    }
    // photometricVisualServoing.cpp lines 400-406
    sI.buildFrom(I);
    vpColVector v = servo.computeControlLaw();
    const double normError = servo.getError().sumSquare();

    std::vector<double> out(6);
    for (unsigned i = 0; i < 6 && i < v.getRows(); ++i) {
      out[i] = v[i];
    }
    return py::make_tuple(out, normError);
  }

  double errorAt(vpImage<unsigned char> &I)
  {
    if (!initialized) {
      throw std::runtime_error("PhotometricServoTask::init() must be called first");
    }
    sI.buildFrom(I);
    servo.computeControlLaw();
    return servo.getError().sumSquare();
  }

private:
  vpFeatureLuminance sI;
  vpFeatureLuminance sId;
  vpServo servo;
  bool initialized = false;
};

static vpImage<unsigned char> numpy_gray_to_vpimage(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr)
{
  auto buf = arr.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("expected H×W uint8 grayscale image");
  }
  const unsigned h = static_cast<unsigned>(buf.shape[0]);
  const unsigned w = static_cast<unsigned>(buf.shape[1]);
  vpImage<unsigned char> I(h, w);
  const auto *src = static_cast<unsigned char *>(buf.ptr);
  for (unsigned r = 0; r < h; ++r) {
    for (unsigned c = 0; c < w; ++c) {
      I[r][c] = src[r * w + c];
    }
  }
  return I;
}

PYBIND11_MODULE(photometric_servo, m)
{
  m.doc() = "ViSP photometricVisualServoing.cpp task (vpFeatureLuminance + vpServo)";

  py::class_<PhotometricServoTask>(m, "PhotometricServoTask")
      .def(py::init<>())
      .def("init", [](PhotometricServoTask &self, py::array_t<uint8_t> Id, double px, double py, double u0, double v0,
                      double Z, double lambda) {
        vpCameraParameters cam(px, py, u0, v0);
        vpImage<unsigned char> id = numpy_gray_to_vpimage(Id);
        self.init(id, cam, Z, lambda);
      })
      .def("step", [](PhotometricServoTask &self, py::array_t<uint8_t> I) {
        vpImage<unsigned char> img = numpy_gray_to_vpimage(I);
        return self.step(img);
      })
      .def("error_at", [](PhotometricServoTask &self, py::array_t<uint8_t> I) {
        vpImage<unsigned char> img = numpy_gray_to_vpimage(I);
        return self.errorAt(img);
      });
}
