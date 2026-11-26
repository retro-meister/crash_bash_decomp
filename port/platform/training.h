#ifndef BASH_TRAINING_H
#define BASH_TRAINING_H

#include <stdint.h>

enum { TRAINING_PLAYERS = 4, TRAINING_BALLS = 5, TRAINING_FRAMES = 3, TRAINING_SAMPLE_VBLANKS = 2 };
enum { TRAINING_CRASHBALL_MODE = 14 };

typedef struct {
    int position[TRAINING_PLAYERS];
    int score[TRAINING_PLAYERS];
    int kick_ready[TRAINING_PLAYERS];
    int balls[TRAINING_BALLS][2];
    int ball_count;
    int round_active;
} TrainingFrame;

typedef struct {
    uint32_t seed;
    int players;
    int difficulty;
} TrainingMatch;

void training_start(int view);
int training_take_match(TrainingMatch *match);
int training_pad(int port, int *buttons);
void training_contact(unsigned player, int kick);
int training_vblank(void);
int training_snapshot(TrainingFrame *frame);

#endif
