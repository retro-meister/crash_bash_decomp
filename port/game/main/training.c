#include "../../platform/training.h"
#include "bash/arkenoid_ovl.h"
#include "bash/main.h"

int training_snapshot(TrainingFrame *frame) {
    NAS *ball;
    int i;

    if (GameMode != TRAINING_CRASHBALL_MODE) {
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
