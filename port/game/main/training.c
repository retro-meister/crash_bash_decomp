#include "../../platform/training.h"
#include "bash/arkenoid_ovl.h"
#include "bash/main.h"
#include "bash/polar_ovl.h"

static void polar_training_snapshot(TrainingFrame *frame) {
    int i;

    frame->ice_radius = PolarIceRadius;
    frame->clock = Clock_Time;
    frame->winner = -1;
    for (i = 0; i < TRAINING_PLAYERS; i++) {
        const HeroInfo *hero = &Heroes[i];
        const EXTengine *engine = hero->hero_eng;
        const Ext20 *polar = FindExtension(hero->hero, 0x20000000);
        int *state = frame->polar[i];

        frame->score[i] = !(hero->hero_status & 0xA000);
        if (hero->hero_status & 0x200) {
            frame->winner = i;
        }
        state[0] = engine->XYZ.x;
        state[1] = engine->XYZ.z;
        state[2] = hero->hero_move ? hero->hero_move->x_vel : 0;
        state[3] = hero->hero_move ? hero->hero_move->z_vel : 0;
        state[4] = hero->hero_move ? hero->hero_move->forward_angle & 0xFFF : 0;
        state[5] = polar ? polar->val : 0;
        state[6] = hero->hero_mode;
        state[7] = frame->score[i];
        state[8] = hero->hero_name;
        state[9] = hero->hero_ai ? hero->hero_ai->AIpad : 0;
    }
}

int training_snapshot(TrainingFrame *frame, int arena) {
    NAS *ball;
    int i;

    if (GameMode != arena) {
        return 0;
    }
    for (i = 0; i < TRAINING_PLAYERS; i++) {
        const HeroInfo *hero = &Heroes[i];

        if (hero->hero_eng == NULL) {
            return 0;
        }
        frame->score[i] = hero->hero_score;
        frame->position[i] = i < 2 ? hero->hero_eng->XYZ.x : hero->hero_eng->XYZ.z;
        frame->kick_ready[i] = hero->hero_attack_timer == 0;
    }
    frame->round_active = WinCounterState == 4;
    if (arena == TRAINING_POLAR_PANIC_MODE) {
        polar_training_snapshot(frame);
        return 1;
    }
    frame->ball_count = 0;
    ball = HeadOfBalls;
    for (i = 0; ball != NULL && i < TRAINING_BALLS; i++, ball = ball->Group.Child) {
        const EXTobject *object = (const EXTobject *)ball->FUX;
        const EXTengine *engine = ball->Engine;

        if (object != NULL && engine != NULL && object->status == 1) {
            int index = frame->ball_count++;

            frame->balls[index][0] = engine->XYZ.x;
            frame->balls[index][1] = engine->XYZ.z;
        }
    }
    return 1;
}
