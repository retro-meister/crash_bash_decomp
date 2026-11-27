#ifndef BASH_TRAINING_H
#define BASH_TRAINING_H

#include <stdint.h>

enum { TRAINING_PLAYERS = 4, TRAINING_BALLS = 5, TRAINING_FRAMES = 3, TRAINING_SAMPLE_VBLANKS = 2 };
enum { TRAINING_CRASHBALL_MODE = 14, TRAINING_POLAR_PANIC_MODE = 19, TRAINING_TINY = 5 };
enum { TRAINING_INPUT, TRAINING_TEACHER, TRAINING_HYBRID, TRAINING_PLANNER };
enum { TRAINING_POLAR_FIELDS = 14, TRAINING_EDGE_RAYS = 8, TRAINING_OBSTACLES = 3 };

typedef struct {
    int position[TRAINING_PLAYERS];
    int score[TRAINING_PLAYERS];
    int kick_ready[TRAINING_PLAYERS];
    int balls[TRAINING_BALLS][2];
    int ball_count;
    int round_active;
    int polar[TRAINING_PLAYERS][TRAINING_POLAR_FIELDS];
    int edge[TRAINING_EDGE_RAYS];
    int obstacles[TRAINING_OBSTACLES][2];
    int obstacle_count, square_held;
    int ice_radius, clock, winner;
} TrainingFrame;

typedef struct {
    uint32_t seed;
    int players;
    int difficulty;
    int arena, hero, teacher;
} TrainingMatch;

void training_start(int view, int arena, int teacher, int repeat);
int training_take_match(TrainingMatch *match);
int training_pad(int port, int *buttons);
void training_contact(unsigned player, int kick);
void training_charge_started(unsigned player);
int training_charge_choice(unsigned player, int angle);
int training_charge_turn(unsigned player);
int training_polar_status(void);
int training_polar_plan(int previous_buttons);
int training_vblank(void);
int training_snapshot(TrainingFrame *frame, int arena);

#endif
