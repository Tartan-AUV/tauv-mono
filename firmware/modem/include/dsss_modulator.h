//
// Created by Gleb Ryabtsev on 7/27/23.
//

#ifndef MODEM_DSSS_MODULATOR_H
#define MODEM_DSSS_MODULATOR_H

#include "main.h"

class DSSSModulator {
public:
    status_t transmit(uint8_t *buf, size_t size);

private:
};
#endif //MODEM_DSSS_MODULATOR_H
