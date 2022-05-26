#include <getopt.h>

struct globalArgs_t {
    int ForwardOrNot;           /* -F option */
    int InversionOrNot;         /* -I option */
    int randomized;             /* --randomize option */
} globalArgs;

static const char *optString = "FIh?";

static const struct option longOpts[] = {
    { "ForwardModeling", no_argument, NULL, 'F' },
    { "Inversion", no_argument, NULL, 'I' },
    { "randomize", no_argument, NULL, 0 },
    { "help", no_argument, NULL, 'h' },
    { NULL, no_argument, NULL, 0 }
};

/* Display program usage, and exit.
 */
void display_usage( void )
{
    puts( "3D_ForwModel_Inversion - 3D ForwardModeling and Inversion of Magnetic Data" );
    puts( "Please input 3D_ForwModel_Inversion -h for help ~");
    /* ... */
    exit( EXIT_FAILURE );
}

void display_help( void )
{
    printf("usage: 3D_ForwModel_Inversion [options]\n\n");
    printf("options:\n");
    printf("  -F     3D forward modeling of magnetic parameters\n");
    printf("  -I     3D inversion of magnetic data (all types)\n");
    printf("  -h     List descriptions of available modules\n\n");
    printf("Caution  -F and -I can not be used simultaneously\n");
}

void print_information( void )
{
    /* ... */
    printf( "ForwardModeling: %d\n", globalArgs.ForwardOrNot );
    printf( "Inversion      : %d\n", globalArgs.InversionOrNot );
    printf( "randomized: %d\n", globalArgs.randomized );
}