#include "training.h"
#include "platform.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int Enabled, PendingMatch;
static int View, Arena, Teacher, FinishVblanks;
static int Buttons[TRAINING_PLAYERS];
static int Contacts[TRAINING_PLAYERS][2];
static TrainingMatch Match;
static TrainingFrame Frames[TRAINING_FRAMES];
static int FrameCount, Started, Vblanks;

static void reply(const char *message) {
    puts(message);
    fflush(stdout);
}

static void read_command(int allow_step) {
    char line[128];

    while (fgets(line, sizeof(line), stdin) != NULL) {
        TrainingMatch next;
        unsigned int buttons[TRAINING_PLAYERS];
        char extra;
        int i, fields;

        if (strchr(line, '\n') == NULL && !feof(stdin)) {
            int c;

            do { c = getchar(); } while (c != '\n' && c != EOF);
            reply("@training {\"error\":\"command too long\"}");
            continue;
        }
        if (strcmp(line, "quit\n") == 0 || strcmp(line, "quit\r\n") == 0 || strcmp(line, "quit") == 0) {
            platform_finish(0);
        }
        next.hero = 0;
        next.arena = Arena;
        next.teacher = Teacher;
        fields = sscanf(line, "reset %" SCNu32 " %d %d %d %c", &next.seed, &next.players, &next.difficulty, &next.hero, &extra);
        if ((fields == 4 || (fields == 3 && sscanf(line, "reset %*u %*d %*d %c", &extra) == EOF)) &&
            next.players >= 1 && next.players <= TRAINING_PLAYERS && next.difficulty >= 0 && next.difficulty <= 2 &&
            (next.hero == 0 || next.hero == TRAINING_TINY)) {
            Match = next;
            PendingMatch = 1;
            FrameCount = 0;
            Started = 0;
            Vblanks = 0;
            memset(Buttons, 0, sizeof(Buttons));
            memset(Contacts, 0, sizeof(Contacts));
            return;
        }
        if (allow_step && sscanf(line, "step %x %x %x %x %c", &buttons[0], &buttons[1], &buttons[2], &buttons[3], &extra) == 4) {
            const unsigned int allowed = PAD_LEFT | PAD_RIGHT | PAD_UP | PAD_DOWN | PAD_R1 | PAD_SQUARE;

            for (i = 0; i < TRAINING_PLAYERS && !(buttons[i] & ~allowed); i++) {
                if (i >= Match.players && buttons[i] != 0) {
                    break;
                }
            }
            if (i == TRAINING_PLAYERS) {
                for (i = 0; i < TRAINING_PLAYERS; i++) {
                    Buttons[i] = (int)buttons[i];
                }
                FrameCount = 0;
                memset(Contacts, 0, sizeof(Contacts));
                return;
            }
        }
        reply(allow_step ? "@training {\"error\":\"expected step, reset or quit\"}"
                         : "@training {\"error\":\"expected reset or quit\"}");
    }
    platform_finish(ferror(stdin) ? 1 : 0);
}

void training_start(int view, int arena, int teacher) {
    Enabled = 1;
    View = view;
    Arena = arena;
    Teacher = teacher;
    reply(arena == TRAINING_POLAR_PANIC_MODE
        ? "@training {\"version\":1,\"mode\":\"Polar Panic\",\"vblanks_per_step\":2}"
        : "@training {\"version\":1,\"mode\":\"Crashball\",\"vblanks_per_step\":6}");
    read_command(0);
}

int training_take_match(TrainingMatch *match) {
    if (!PendingMatch) {
        return 0;
    }
    *match = Match;
    PendingMatch = 0;
    return 1;
}

int training_pad(int port, int *buttons) {
    if (!Enabled) {
        return 0;
    }
    *buttons = port >= 0 && port < Match.players ? Buttons[port] : -1;
    return 1;
}

void training_contact(unsigned player, int kick) {
    if (Enabled && Started && player < TRAINING_PLAYERS) {
        Contacts[player][kick != 0]++;
    }
}

static void write_frames(int count, int done) {
    int i, j;

    printf("@training {\"done\":%s,\"frames\":[", done ? "true" : "false");
    for (i = 0; i < count; i++) {
        const TrainingFrame *f = &Frames[i];

        if (Arena == TRAINING_POLAR_PANIC_MODE) {
            printf("%s{\"score\":[%d,%d,%d,%d],\"ice_radius\":%d,\"clock\":%d,\"winner\":%d,\"polar\":[",
                   i == 0 ? "" : ",", f->score[0], f->score[1], f->score[2], f->score[3],
                   f->ice_radius, f->clock, f->winner);
            for (j = 0; j < TRAINING_PLAYERS; j++) {
                const int *p = f->polar[j];
                printf("%s[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]", j == 0 ? "" : ",",
                       p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
            }
            printf("]}");
            continue;
        }
        printf("%s{\"position\":[%d,%d,%d,%d],\"score\":[%d,%d,%d,%d],\"kick_ready\":[%d,%d,%d,%d],\"balls\":[",
               i == 0 ? "" : ",", f->position[0], f->position[1], f->position[2], f->position[3],
               f->score[0], f->score[1], f->score[2], f->score[3],
               f->kick_ready[0], f->kick_ready[1], f->kick_ready[2], f->kick_ready[3]);
        for (j = 0; j < f->ball_count; j++) {
            printf("%s[%d,%d]", j == 0 ? "" : ",", f->balls[j][0], f->balls[j][1]);
        }
        printf("]}");
    }
    printf("],\"touches\":[%d,%d,%d,%d],\"kicks\":[%d,%d,%d,%d]}",
           Contacts[0][0], Contacts[1][0], Contacts[2][0], Contacts[3][0],
           Contacts[0][1], Contacts[1][1], Contacts[2][1], Contacts[3][1]);
    reply("");
}

int training_vblank(void) {
    TrainingFrame *frame;
    int alive = 0, done, i;

    if (FinishVblanks != 0) {
        return --FinishVblanks == 0;
    }
    if (!Enabled || PendingMatch || ++Vblanks % TRAINING_SAMPLE_VBLANKS != 0) {
        return 0;
    }
    frame = &Frames[FrameCount];
    if (!Started && Vblanks >= 1800) {
        reply("@training {\"error\":\"Arena did not reach the start of the round\"}");
        platform_finish(1);
    }
    if (!training_snapshot(frame, Arena)) {
        if (Started) {
            reply("@training {\"error\":\"Arena left the round unexpectedly\"}");
            platform_finish(1);
        }
        return 0;
    }
    for (i = 0; i < TRAINING_PLAYERS; i++) {
        alive += frame->score[i] > 0;
    }
    if (!Started && (!frame->round_active || alive < 2)) {
        return 0;
    }
    Started = 1;
    done = alive < 2 || !frame->round_active;
    if (++FrameCount == (Arena == TRAINING_POLAR_PANIC_MODE ? 1 : TRAINING_FRAMES) || done) {
        write_frames(FrameCount, done);
        if (View && done) {
            FinishVblanks = 120;
            memset(Buttons, 0, sizeof(Buttons));
            return 0;
        }
        read_command(!done);
        if (PendingMatch) {
            platform_restart(0);
        }
    }
    return 0;
}
