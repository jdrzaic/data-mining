#include "constants.h"
#include "matrix.h"
#include "page_rank.h"


#include <argp.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>


using namespace std;


const char *argp_program_version = "pr 0.1";
const char *argp_program_bug_address = "<jdrzaic@student.math.hr>";
static char doc[] =
    "pr --- a minimal PageRank calculator\n"
    "Reads a graph matrix from FILE stored in MatrixMarket COO format "
    "and computes the PageRank vector of the graph.\n"
    "The default options are:\n\tpr -i1 -s2 -t1e-6 -a0.0 FILE\n";
static char args_doc[] = "FILE";
static struct argp_option options[] = {
    {"iterations"  , 'i', "N", 0, "maximal number of iterations"},
    {"subspace-dim", 's', "N", 0, "dimension of the Krylov subspace"},
    {"tolerance"   , 't', "N", 0, "desired accuracy of the solution vector"},
    {"alpha"       , 'a', "N", 0, "random jump probability"},
    {"power"       , 'p',   0, 0, "use power method instead of Arnoldi"},
    {0}
};


struct arguments {
    arguments()
        : max_iters(1), subspace_dim(2), tol(1e-6), alpha(0.0),
          alg(AlgArnoldi), filename(nullptr) {}
    int max_iters;
    int subspace_dim;
    real tol;
    real alpha;
    enum {
        AlgPower,
        AlgArnoldi
    } alg;
    const char *filename;
};


static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    arguments *args = (arguments*)state->input;
    switch (key) {
    case 'i':
        args->max_iters = std::atoi(arg);
        break;
    case 's':
        args->subspace_dim = std::atoi(arg);
        break;
    case 't':
        args->tol = std::atof(arg);
        break;
    case 'a':
        args->alpha = std::atof(arg);
        break;
    case 'p':
        args->alg = arguments::AlgPower;
        break;
    case ARGP_KEY_ARG:
        if (state->arg_num >= 1) {
            // too many arguments
            argp_usage(state);
        }
        args->filename = arg;
        break;
    case ARGP_KEY_END:
        if (state->arg_num < 1) {
            // too few arguments
            argp_usage(state);
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}
static struct argp argp = { options, parse_opt, args_doc, doc };


int main(int argc, char *argv[])
{
    arguments args;
    argp_parse(&argp, argc, argv, 0, nullptr, &args);
    context ctx;
    ctx.create();
    cout.precision(6);
    cout << scientific << showpos;
    ifstream fin(args.filename);
    CsrMatrix<real, DataHost> A;
    read_mtx(fin, &A);
    Array<real, DataHost> x;
    x.init(A.n_rows);
    for (int i = 0; i < x.size; ++i) {
        x[i] = 1.0;
    }

    if (args.alg == arguments::AlgArnoldi) {
        get_pagerank_vector_arnoldi(
                ctx, &A, args.alpha, args.subspace_dim, &x, args.tol,
                args.max_iters);
    } else {
        get_pagerank_vector_power(
                ctx, &A, args.alpha, &x, args.tol, args.max_iters);
    }

    std::cout << "x = [\n" << x << "];" << std::endl;

    ctx.destroy();
    return 0;
}

