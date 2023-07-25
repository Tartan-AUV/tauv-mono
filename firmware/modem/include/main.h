//
// Created by Gleb Ryabtsev on 7/25/23.
//

#ifndef MODEM_MAIN_H
#define MODEM_MAIN_H

#define ONE_SECOND_NS 1'000'000'000

#define PIN_TX 18
#define PIN_TX_EN 20
#define PIN_DBG_1 10
#define PIN_DBG_2 11

#define RAW_BUF_SIZE 100
#define SDFT_UNDERSAMPLING_RATIO 8
#define SDFT_BUF_SIZE (RAW_BUF_SIZE / SDFT_UNDERSAMPLING_RATIO)


#endif //MODEM_MAIN_H
