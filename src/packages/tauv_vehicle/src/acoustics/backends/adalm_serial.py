import libm2k
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class AdalmSerialBackend:

    def __init__(self):
        self._sample_frequency = 1000000
        self._sample_size = 1000

        self._ctx0: Optional[libm2k.Context] = None
        self._ctx1: Optional[libm2k.Context] = None
        self._ain0: Optional[libm2k.M2kAnalogIn] = None
        self._ain1: Optional[libm2k.M2kAnalogIn] = None
        self._trig0: Optional[libm2k.M2kHardwareTrigger] = None
        self._trig1: Optional[libm2k.M2kHardwareTrigger] = None

    def open(self):
        ctxs = libm2k.getAllContexts()
        print(ctxs)
        assert(len(ctxs) == 2)

        self._ctx0 = libm2k.m2kOpen(ctxs[0])
        self._ctx1 = libm2k.m2kOpen(ctxs[1])

        self._ain0, self._trig0 = self._configure_ctx(self._ctx0)
        self._ain1, self._trig1 = self._configure_ctx(self._ctx1)

        self._configure_source_trig(self._trig0)
        self._configure_sink_trig(self._trig1)

    def close(self):
        if self._ctx0 is not None:
            libm2k.contextClose(self._ctx0)
            self._ctx0 = None
            self._ain0 = None
            self._trig0 = None

        if self._ctx1 is not None:
            libm2k.contextClose(self._ctx1)
            self._ctx1 = None
            self._ain1 = None
            self._trig1 = None

    def sample(self) -> np.array:
        self._ain0.startAcquisition(self._sample_size)
        self._ain1.startAcquisition(self._sample_size)

        data0 = self._ain0.getSamples(self._sample_size)
        data1 = self._ain1.getSamples(self._sample_size)

        self._ain0.stopAcquisition()
        self._ain1.stopAcquisition()

        return np.array([data0[0], data0[1], data1[0], data1[1]])

    def _configure_ctx(self, ctx: libm2k.Context) -> (libm2k.M2kAnalogIn, libm2k.M2kHardwareTrigger):
        ctx.setTimeout(500)

        ain = ctx.getAnalogIn()
        trig = ain.getTrigger()

        ain.reset()
        ctx.calibrateADC()

        ain.enableChannel(0, True)
        ain.enableChannel(1, True)
        ain.setSampleRate(self._sample_frequency)
        ain.setRange(0, libm2k.PLUS_MINUS_2_5V)
        ain.setRange(1, libm2k.PLUS_MINUS_2_5V)

        return ain, trig

    def _configure_source_trig(self, trig: libm2k.M2kHardwareTrigger):
        trig.setAnalogMode(0, libm2k.ANALOG)
        trig.setAnalogExternalOutSelect(libm2k.SELECT_ANALOG_IN)
        trig.setAnalogSource(0)
        trig.setAnalogCondition(0, libm2k.RISING_EDGE_ANALOG)
        trig.setAnalogLevel(0, 0.5)
        trig.setAnalogDelay(0)

    def _configure_sink_trig(self, trig: libm2k.M2kHardwareTrigger):
        trig.setAnalogMode(0, libm2k.EXTERNAL)
        trig.setAnalogSource(0)
        trig.setAnalogExternalCondition(0, libm2k.RISING_EDGE_ANALOG)
        trig.setAnalogLevel(0, 0.5)
        trig.setAnalogDelay(0)

if __name__ == "__main__":
    a = AdalmSerialBackend()
    a.open()
    try:
        samples = a.sample()
        print(samples)

        time = np.arange(0, a._sample_size) / a._sample_frequency

        plt.plot(time, samples[0])
        plt.plot(time, samples[1])
        plt.plot(time, samples[2])
        plt.plot(time, samples[3])
        plt.show()
    except Exception as e:
        print(e) 
    a.close()