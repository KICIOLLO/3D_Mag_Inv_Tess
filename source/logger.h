#include <stdio.h>
#include <stdarg.h>
#include <time.h>
/*
Functions to set up logging.

Examples
--------

Logging to stderr:

    #include "logger.h"

    void my_func(){
        log_info("From my_func!\n");
    }

    int main(){
        // Enable logging to stderr in debug level
        // will only print messages of level DEBUG or higher
        log_init(LOG_DEBUG);
        log_debug("debug line. The code is %d", LOG_DEBUG);
        log_info("info line. The code is %d", LOG_INFO);
        log_warning("warning line. The code is %d", LOG_WARNING);
        log_error("error line. The code is %d", LOG_ERROR);
        my_func();
        return 0;
    }

will print:

    DEBUG: debug line. The code is 0
    info line. The code is 1
    WARNING: warning line. The code is 2
    ERROR: error line. The code is 3
    From my_func!

If function log_init() is not called than logging to stderr is disabled and no
messages will be printed.

Logging to a file:

    #include <stdio.h>
    #include "logger.h"

    void my_func(){
        log_info("From my_func!\n");
        log_debug("Should not appear in log file\n");
    }

    int main(){
        // Enable logging to file "log.txt" in info level
        // will not print DEBUG level messages
        // since log_init was not called, there is no logging to stderr
        FILE *logfile = fopen("log.txt", "w");
        log_tofile(logfile, LOG_INFO);
        log_debug("debug line. The code is %d", LOG_DEBUG);
        log_info("info line. The code is %d", LOG_INFO);
        log_warning("warning line. The code is %d", LOG_WARNING);
        log_error("error line. The code is %d", LOG_ERROR);
        my_func();
        return 0;
    }

File log.txt will look like:

    info line. The code is 1
    WARNING: warning line. The code is 2
    ERROR: error line. The code is 3
    From my_func!

Note that you can combine loggin to stderr and to a file with different
levels in the same program.
*/

#ifndef _TESSEROIDS_LOGGER_H_
#define _TESSEROIDS_LOGGER_H_

/* Needed for definition of FILE */
#include <stdio.h>


/** Logging level for debug messages */
#define LOG_DEBUG   1
/** Logging level for general information */
#define LOG_INFO    2
/** Logging level for warning messages */
#define LOG_WARNING 3
/** Logging level for error messages */
#define LOG_ERROR   4


/** Keep the information on the global logger */
typedef struct logger_struct
{
    int level; /**< level of logging */
    int filelogging; /**< flag to know wether loggint to a file is enabled */
    int file_level; /**< logging level for the file */
    FILE *logfile; /**< file to log to */

} LOGGER;


/** Global logger struct. Only declare in the main program! */
//LOGGER logger;


/** Setup logging to stderr.

@param level level of logging to be made. Can be one of:
    - LOG_DEBUG
    - LOG_INFO
    - LOG_WARNING
    - LOG_ERROR
*/
void log_init(int level);


/** Set logging to a file.

@param logfile FILE pointer to the already open file to log to.
@param level level of logging to be made to the file. Can be one of:
    - LOG_DEBUG
    - LOG_INFO
    - LOG_WARNING
    - LOG_ERROR
*/
void log_tofile(FILE *logfile, int level);


/** Log a message at debug level.

Pass parameters in the same format as printf()

Prints a newline at the end.
*/
void log_debug(const char *fmt, ...);


/** Log a message at info level.

Pass parameters in the same format as printf()

Does not print "INFO: " in front of the message when logging

Prints a newline at the end.
*/
void log_info(const char *fmt, ...);


/** Log a message at warning level.

Pass parameters in the same format as printf()

Prints a newline at the end.
*/
void log_warning(const char *fmt, ...);


/** Log a message at error level.

Pass parameters in the same format as printf()

Prints a newline at the end.
*/
void log_error(const char *fmt, ...);

/* Initialize the logger so that it doesn't print by default */
LOGGER logger = { 100, 0, 100, NULL };


/* Setup logging to stderr.*/
void log_init(int level)
{
	logger.level = level;
}


/* Set logging to a file. */
void log_tofile(FILE *logfile, int level)
{
	logger.filelogging = 1;
	logger.logfile = logfile;
	logger.file_level = level;
}


/* Log a message at debug level */
void log_debug(const char *fmt, ...)
{
	char msg[10000];
	va_list args;
	va_start(args, fmt);
	vsprintf(msg, fmt, args);
	va_end(args);

	if (logger.level <= LOG_DEBUG)
	{
		fprintf(stderr, "DEBUG: %s\n", msg);
	}

	if (logger.filelogging && (logger.file_level <= LOG_DEBUG))
	{
		fprintf(logger.logfile, "DEBUG: %s\n", msg);
	}
}


/* Log a message at info level */
void log_info(const char *fmt, ...)
{
	char msg[10000];
	va_list args;
	va_start(args, fmt);
	vsprintf(msg, fmt, args);
	va_end(args);

	if (logger.level <= LOG_INFO)
	{
		fprintf(stderr, "%s\n", msg);
	}

	if (logger.filelogging && logger.file_level <= LOG_INFO)
	{
		fprintf(logger.logfile, "%s\n", msg);
	}
}


/* Log a message at warning level */
void log_warning(const char *fmt, ...)
{
	char msg[10000];
	va_list args;
	va_start(args, fmt);
	vsprintf(msg, fmt, args);
	va_end(args);

	if (logger.level <= LOG_WARNING)
	{
		fprintf(stderr, "WARNING: %s\n", msg);
	}

	if (logger.filelogging && logger.file_level <= LOG_WARNING)
	{
		fprintf(logger.logfile, "WARNING: %s\n", msg);
	}
}


/* Log a message at error level */
void log_error(const char *fmt, ...)
{
	char msg[10000];
	va_list args;
	va_start(args, fmt);
	vsprintf(msg, fmt, args);
	va_end(args);

	if (logger.level <= LOG_ERROR)
	{
		fprintf(stderr, "ERROR: %s\n", msg);
	}

	if (logger.filelogging && logger.file_level <= LOG_ERROR)
	{
		fprintf(logger.logfile, "ERROR: %s\n", msg);
	}
}
#endif
