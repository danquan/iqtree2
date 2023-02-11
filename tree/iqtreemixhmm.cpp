//
//  iqtreemixhmm.cpp
//  tree
//
//  Created by Thomas Wong on 19/01/23.
//

#include "iqtreemixhmm.h"

IQTreeMixHmm::IQTreeMixHmm() : IQTreeMix(), PhyloHmm() {
}

IQTreeMixHmm::IQTreeMixHmm(Params &params, Alignment *aln, vector<IQTree*> &trees) : IQTreeMix(params, aln, trees), PhyloHmm(getAlnNSite(), trees.size()) {
    optimTree = -1;
    optimBranchGrp = -1;
}

// initialize the model
void IQTreeMixHmm::initializeModel(Params &params, string model_name, ModelsBlock *models_block) {
    IQTreeMix::initializeModel(params, model_name, models_block);
    size_t i;
    
    // for all the unlinked substitution models,
    // set all trees to this tree
    // if (!isLinkModel) {
    //    for (i=0; i<models.size(); i++) {
    //        models[i]->setTree(this);
    //    }
    // }
    
    // handle the linked or unlinked site rate(s)
    site_rates.clear();
    site_rate_trees.clear();
    if (anySiteRate) {
        if (isLinkSiteRate) {
            // for linked site rate model, set all site rates to site_rates[0]
            site_rates.push_back(at(0)->getModelFactory()->site_rate);
            for (i=1; i<ntree; i++) {
                at(i)->getModelFactory()->site_rate = site_rates[0];
                at(i)->setRate(site_rates[0]);
            }
            // for linked site rate model, set their trees to this tree
            for (i=0; i<site_rates.size(); i++) {
                site_rates[i]->setTree(this);
            }
            for (i=0; i<ntree; i++) {
                site_rate_trees.push_back(this);
            }
        } else {
            for (i=0; i<ntree; i++) {
                site_rates.push_back(at(i)->getModelFactory()->site_rate);
            }
            // for unlinked site rate models, set their trees to its own tree
            for (i=0; i<site_rates.size(); i++) {
                site_rates[i]->setTree(at(i));
            }
            for (i=0; i<ntree; i++) {
                site_rate_trees.push_back(at(i));
            }
        }
    }
    
    // edge-length-restricted model is not appropriate for this HMM model
    if (isEdgeLenRestrict)
        outError("Edge-length-restricted model is inappropriate for HMM model. Use +T instead of +TR.");
    
    // build the branch ID
    computeBranchID();
    if (verbose_mode >= VB_MED) {
        showBranchGrp();
    }
}

// obtain the log-likelihoods for every site and every tree
// output site_like_cat[i * ntree + j] : log-likelihood of site nsite-i-1 and tree j
void IQTreeMixHmm::computeLogLikelihoodSiteTree(int updateTree) {
    
    if (updateTree > -1) {
        // only update a single tree
        computeLogLikelihoodSingleTree(updateTree);
    } else {
        // update all trees
        // compute likelihood for each tree
        for (int t=0; t<ntree; t++) {
            computeLogLikelihoodSingleTree(t);
        }
    }

    // reorganize the array
    // #pragma omp parallel for schedule(static) num_threads(num_threads) if (num_threads > 1)
    for (int j = 0; j < ntree; j++) {
        int k = nsite;
        int l = j;
        double* ptn_lh_arr = _ptn_like_cat + nptn * j;
        for (int i = 0; i < nsite; i++) {
            k--;
            int ptn = aln->getPatternID(k);
            // ptn_like_cat[i * ntree + j] = log-likelihood of site nsite-i-1 and tree j
            site_like_cat[l] = ptn_lh_arr[ptn];
            l += ntree;
        }
    }
}

double IQTreeMixHmm::computeLikelihood(double *pattern_lh, bool save_log_value) {
    computeLogLikelihoodSiteTree(optimTree);
    return computeBackLike();
}

double IQTreeMixHmm::optimizeBranchGroup(int branchgrp, double gradient_epsilon) {
    double score;
    int ndim;
    // for dimension = 1
    double len;
    // for dimension > 1

    optimTree = -1;
    optimBranchGrp = branchgrp;
    ndim = getNDim();
    if (ndim == 1) {
        len = setSingleVariable();
        double negative_lh;
        double ferror,optx;
        if (verbose_mode >= VB_MED) {
            cout << "[IQTreeMixHmm::optimizeBranchGroup before] branchgrp = " << branchgrp << " single-variable = (" << setprecision(10) << len << ") ndim = 1" << endl;
        }
        optx = minimizeOneDimen(params->min_branch_length, len, params->max_branch_length, gradient_epsilon, &negative_lh, &ferror);
        getSingleVariable(optx);
        if (verbose_mode >= VB_MED) {
            cout << "[IQTreeMixHmm::optimizeBranchGroup after] branchgrp = " << branchgrp << " single-variable = (" << setprecision(10) << optx << ") ndim = 1" << endl;
        }
        score = computeLikelihood();
    } else if (ndim > 1) {
        double* variables = new double[ndim + 1];
        double* upper_bound = new double[ndim + 1];
        double* lower_bound = new double[ndim + 1];
        bool* bound_check = new bool[ndim + 1];
        setBounds(lower_bound, upper_bound, bound_check);
        setVariables(variables);
        if (verbose_mode >= VB_MED) {
            cout << "[IQTreeMixHmm::optimizeBranchGroup before] branchgrp = " << branchgrp << " variables = (";
            for (int k = 1; k <= ndim; k++) {
                if (k>1)
                    cout << ",";
                cout << setprecision(10) << variables[k];
            }
            cout << ") ndim = " << ndim << endl;
        }
        score = -minimizeMultiDimen(variables, ndim, lower_bound, upper_bound, bound_check, gradient_epsilon);
        getVariables(variables);
        if (verbose_mode >= VB_MED) {
            cout << "[IQTreeMixHmm::optimizeBranchGroup after] branchgrp = " << branchgrp << " variables = (";
            for (int k = 1; k <= ndim; k++) {
                if (k>1)
                    cout << ",";
                cout << setprecision(10) << variables[k];
            }
            cout << ") ndim = " << ndim << endl;
        }
        delete[] variables;
        delete[] upper_bound;
        delete[] lower_bound;
        delete[] bound_check;
    } else {
        optimBranchGrp = -1;
        score = computeLikelihood();
    }
    optimBranchGrp = -1;
    return score;
}

double IQTreeMixHmm::optimizeAllBranchLensByBFGS(double gradient_epsilon, double logl_epsilon, int maxsteps) {
    double score, pre_score, step;
    
    // collect the branch lengths of the tree
    getAllBranchLengths();
    score = computeLikelihood();
    step = 0;
    do {
        pre_score = score;
        step++;
        for (int i = 0; i < branch_group.size(); i++) {
            score = optimizeBranchGroup(i, gradient_epsilon);
            cout << ".. Current HMM log-likelihood: " << score << endl;
        }
    } while (score - pre_score > logl_epsilon && step < maxsteps);
    return score;
}

double IQTreeMixHmm::optimizeAllBranches(int my_iterations, double tolerance, int maxNRStep) {
    double score;
    computeFreqArray();
    for (int i = 0; i < ntree; i++) {
        IQTreeMix::optimizeAllBranchesOneTree(i, my_iterations, tolerance, maxNRStep);
    }
    score = computeLikelihood();
    if (verbose_mode >= VB_MED)
        cout << "after optimizing branches, HMM likelihood = " << score << endl;
    return score;
}

double IQTreeMixHmm::optimizeAllSubstModels(double gradient_epsilon) {
    double score;
    if (isLinkModel) {
        // for linked subsitution model
        // use BFGS method
        models[0]->optimizeParameters(gradient_epsilon);
    } else {
        // for unlinked subsitution model
        // use EM method
        computeFreqArray();
        for (int i = 0; i < ntree; i++) {
            models[i]->optimizeParameters(gradient_epsilon);
        }
    }
    score = computeLikelihood();
    if (verbose_mode >= VB_MED)
        cout << "after optimizing subsitution model, HMM likelihood = " << score << endl;
    return score;
}

double IQTreeMixHmm::optimizeAllRHASModels(double gradient_epsilon, double score) {
    if (anySiteRate) {
        if (isLinkSiteRate) {
            // for linked RHAS model
            // use BFGS method
            site_rates[0]->optimizeParameters(gradient_epsilon);
        } else {
            // for unlinked RHAS model
            // use EM method
            computeFreqArray();
            for (int i = 0; i < ntree; i++) {
                site_rates[i]->optimizeParameters(gradient_epsilon);
            }
        }
        score = computeLikelihood();
        if (verbose_mode >= VB_MED)
            cout << "after optimizing RHAS model, HMM likelihood = " << score << endl;
    }
    return score;
}

void IQTreeMixHmm::startCheckpoint() {
    checkpoint->startStruct("IQTreeMixHmm" + convertIntToString(size()));
}

// ------------------------------------------------------------------

string IQTreeMixHmm::optimizeModelParameters(bool printInfo, double logl_epsilon) {
    size_t i, ptn;
    int step, n, m, substep1, nsubstep1, nsubstep1_start, nsubstep1_max, nsubstep2_start, nsubstep2_max, substep2, nsubstep2, substep2_tot;
    double curr_epsilon;
    double prev_score, prev_score1, prev_score2, score, t_score;
    double gradient_epsilon = 0.0001;
    PhyloTree *ptree;
    
    // the edges with the same partition among the trees are initialized as the same length
    setAvgLenEachBranchGrp();

    cout << setprecision(5) << "Estimate model parameters (epsilon = " << logl_epsilon << ")" << endl;
    
    // minimum value of edge length
    if (verbose_mode >= VB_MED) {
        cout << "Minimum value of edge length is set to: " << setprecision(10) << params->min_branch_length << endl;
    }
    
    score = computeLikelihood();
    
    cout << "1. Initial HMM log-likelihood: " << score << endl;
    
    prev_score = score;

    for (step = 0; step < optimize_steps; step++) {
        
        // optimize tree branches
        score = optimizeAllBranches();

        // optimize all subsitution models
        score = optimizeAllSubstModels(gradient_epsilon);

        // optimize all site rate models
        score = optimizeAllRHASModels(gradient_epsilon, score);

        // optimize transition matrix and prob array
        score = PhyloHmm::optimizeParameters(gradient_epsilon);

        cout << step+2 << ". Current HMM log-likelihood: " << score << endl;
        if (score < prev_score + logl_epsilon) {
            // converged
            break;
        }

//        if (verbose_mode >= VB_MED) {
//            computeMaxPath();
//            showSiteCatMaxLike(cout);
//        }

        prev_score = score;
    }

    backLogLike = score;
    setCurScore(score);
    stop_rule.setCurIt(step);
    computeMaxPath();

    return getTreeString();
}

void IQTreeMixHmm::setNumThreads(int num_threads) {

    PhyloTree::setNumThreads(num_threads);

    for (size_t i = 0; i < size(); i++)
        at(i)->setNumThreads(num_threads);
}

/**
    test the best number of threads
*/
int IQTreeMixHmm::testNumThreads() {
    int bestNThres = at(0)->testNumThreads();
    setNumThreads(bestNThres);
    return bestNThres;
}

int IQTreeMixHmm::getNParameters() {
    int df = 0;
    int k;
    size_t i;
    
    if (verbose_mode >= VB_MED)
        cout << endl << "Number of parameters:" << endl;
    for (i=0; i<models.size(); i++) {
        k = models[i]->getNDim() + models[i]->getNDimFreq();
        if (verbose_mode >= VB_MED) {
            if (models.size() == 1)
                cout << " linked model : " << k << endl;
            else
                cout << " model " << i+1 << " : " << k << endl;
        }
        df += k;
    }
    for (i=0; i<site_rates.size(); i++) {
        if (verbose_mode >= VB_MED) {
            if (site_rates.size() == 1)
                cout << " linked site rate : " << site_rates[i]->getNDim() << endl;
            else
                cout << " siterate " << i+1 << " : " << site_rates[i]->getNDim() << endl;
        }
        df += site_rates[i]->getNDim();
    }
    // for branch parameters
    if (params->fixed_branch_length != BRLEN_FIX) {
        for (i=0; i<size(); i++) {
            k = at(i)->getNBranchParameters(BRLEN_OPTIMIZE);
            if (verbose_mode >= VB_MED)
                cout << " branches of tree " << i+1 << " : " << k << endl;
            df += k;
        }
    }
    // for transition matrix
    if (verbose_mode >= VB_MED)
        cout << " transition matrix : " << modelHmm->getNParameters() << endl;
    df += modelHmm->getNParameters();
    // for probability array
    if (verbose_mode >= VB_MED)
        cout << " probability array : " << ntree - 1 << endl;
    df += ntree - 1;

    if (verbose_mode >= VB_MED)
        cout << " == Total : " << df << " == " << endl << endl;
    return df;
}

// print out all the results to a file
void IQTreeMixHmm::printResults(const char *filename) {
    
    size_t i, j;
    ofstream out;
    out.open(filename);
    
    // report the estimated HMM parameters
    showParameters(out);
    out << endl;
    
    // show the assignment of the categories along sites with max likelihood
    showSiteCatMaxLike(out);
    
    out.close();
}

// print out the HMM estimated parameters
void IQTreeMixHmm::showParameters(ostream& out) {
    size_t i, j;
    modelHmm->showParameters(out);
    out << endl;
    out << "Estimated HMM probabilities :" << endl;
    for (i = 0; i < ntree; i++) {
        if (i > 0)
            out << "\t";
        out << fixed << setprecision(5) << prob[i];
    }
    out << endl << endl;
    
    out << "BEST HMM SCORE FOUND :" << fixed << setprecision(5) << backLogLike << endl;
}

// compute the log-likelihoods for a single tree t
void IQTreeMixHmm::computeLogLikelihoodSingleTree(int t) {
    double* pattern_lh_tree = _ptn_like_cat + nptn * t;
    // save the site rate's tree
    PhyloTree* ptree = at(t)->getRate()->getTree();
    // set the tree t as the site rate's tree
    // and compute the likelihood values
    at(t)->getRate()->setTree(at(t));
    at(t)->initializeAllPartialLh();
    at(t)->clearAllPartialLH();
    at(t)->computeLikelihood(pattern_lh_tree, true); // get the log-likelihood values
    // set back the previous site rate's tree
    at(t)->getRate()->setTree(ptree);
}

// get the branch lengths of all trees to the variable allbranchlens
void IQTreeMixHmm::getAllBranchLengths() {
    if (allbranchlens.size() < ntree)
        allbranchlens.resize(ntree);
    for (size_t i=0; i<ntree; i++)
        at(i)->saveBranchLengths(allbranchlens[i]);
}

// set the branch lengths of all trees from the variable allbranchlens
void IQTreeMixHmm::setAllBranchLengths() {
    for (size_t i=0; i<ntree; i++)
        at(i)->restoreBranchLengths(allbranchlens[i]);
}

// show the branch lengths of all trees
void IQTreeMixHmm::showAllBranchLengths() {
    getAllBranchLengths();
    for (size_t i=0; i<ntree; i++) {
        cout << "The branch lengths of tree " << i+1 << endl;
        for (size_t j=0; j<allbranchlens[i].size(); j++) {
            if (j>0)
                cout << ", ";
            cout << allbranchlens[i].at(j);
        }
        cout << endl;
    }

}

//--------------------------------------
// optimization of branch lengths
//--------------------------------------

// the following three functions are for dimension = 1
double IQTreeMixHmm::computeFunction(double x) {
    getSingleVariable(x);
    return -computeLikelihood();
}

double IQTreeMixHmm::setSingleVariable() {
    // get the value from branch lengths
    int ndim, i;
    int treeidx, branchidx;
    double x = 0.0;
    // collect the branch lengths of the tree
    getAllBranchLengths();
    ndim = getNDim();
    if (ndim > 0) {
        treeidx = branch_group[optimBranchGrp].at(0) / nbranch;
        branchidx = branch_group[optimBranchGrp].at(0) % nbranch;
        if (treeidx < ntree && branchidx < allbranchlens[treeidx].size())
            x = allbranchlens[treeidx].at(branchidx);
        else
            cout << "[IQTreeMixHmm::setSingleVariable] Error occurs! treeidx = " << treeidx << ", branchidx = " << branchidx << endl;
    } else {
        cout << "[IQTreeMixHmm::setSingleVariable] Error occurs! ndim = " << ndim << endl;
    }
    return x;
}

void IQTreeMixHmm::getSingleVariable(double x) {
    // save the values to branch lengths
    int ndim, i;
    int treeidx, branchidx;
    // collect the branch lengths of the tree
    getAllBranchLengths();
    ndim = getNDim();
    if (ndim > 0) {
        treeidx = branch_group[optimBranchGrp].at(0) / nbranch;
        branchidx = branch_group[optimBranchGrp].at(0) % nbranch;
        if (treeidx < ntree && branchidx < allbranchlens[treeidx].size())
            allbranchlens[treeidx].at(branchidx) = x;
        else
            cout << "[IQTreeMixHmm::getSingleVariable] Error occurs! treeidx = " << treeidx << ", branchidx = " << branchidx << endl;
    } else {
        cout << "[IQTreeMixHmm::getSingleVariable] Error occurs! ndim = " << ndim << endl;
    }
    setAllBranchLengths();
}

// the following four functions are for dimension > 1
double IQTreeMixHmm::targetFunk(double x[]) {
    getVariables(x);
    return -computeLikelihood();
}

void IQTreeMixHmm::setVariables(double *variables) {
    // copy the values from branch lengths
    int ndim, i;
    int treeidx, branchidx;
    // collect the branch lengths of the tree
    getAllBranchLengths();
    ndim = getNDim();
    for (i=0; i<ndim; i++) {
        treeidx = branch_group[optimBranchGrp].at(i) / nbranch;
        branchidx = branch_group[optimBranchGrp].at(i) % nbranch;
        if (treeidx < ntree && branchidx < allbranchlens[treeidx].size())
            variables[i+1] = allbranchlens[treeidx].at(branchidx);
        else
            cout << "[IQTreeMixHmm::setVariables] Error occurs! treeidx = " << treeidx << ", branchidx = " << branchidx << endl;
    }
    if (ndim == 0) {
        cout << "[IQTreeMixHmm::setVariables] Error occurs! ndim = " << ndim << endl;
    }
}


void IQTreeMixHmm::getVariables(double *variables) {
    // save the values to branch lengths
    int ndim, i;
    int treeidx, branchidx;
    // collect the branch lengths of the tree
    getAllBranchLengths();
    ndim = getNDim();
    for (i=0; i<ndim; i++) {
        treeidx = branch_group[optimBranchGrp].at(i) / nbranch;
        branchidx = branch_group[optimBranchGrp].at(i) % nbranch;
        if (treeidx < ntree && branchidx < allbranchlens[treeidx].size())
            allbranchlens[treeidx].at(branchidx) = variables[i+1];
        else
            cout << "[IQTreeMixHmm::getVariables] Error occurs! treeidx = " << treeidx << ", branchidx = " << branchidx << endl;
    }
    if (ndim == 0) {
        cout << "[IQTreeMixHmm::getVariables] Error occurs! ndim = " << ndim << endl;
    }
    setAllBranchLengths();
}

void IQTreeMixHmm::setBounds(double *lower_bound, double *upper_bound, bool *bound_check) {
    int ndim, i;
    ndim = getNDim();
    if (verbose_mode >= VB_MED) {
        cout << "[IQTreeMixHmm::setBounds] optimBranchGrp = " << optimBranchGrp << ", ndim = " << ndim << endl;
    }
    for (i = 1; i <= ndim; i++) {
        lower_bound[i] = params->min_branch_length;
        upper_bound[i] = params->max_branch_length;
        bound_check[i] = false;
    }
    if (ndim == 0) {
        cout << "[IQTreeMixHmm::setBounds] Error occurs! ndim = " << ndim << endl;
    }
}


int IQTreeMixHmm::getNDim() {
    if (optimBranchGrp >= 0 && optimBranchGrp < branch_group.size()) {
        return branch_group[optimBranchGrp].size();
    } else {
        return 0;
    }
}

void IQTreeMixHmm::showBranchGrp() {
    cout << "Branch Group:" << endl;
    for (size_t i=0; i<branch_group.size(); i++) {
        cout << "  Grp " << i << endl;
        for (size_t j=0; j<branch_group[i].size(); j++) {
            if (j > 0)
                cout << ", ";
            else
                cout << "    ";
            cout << branch_group[i].at(j);
        }
        cout << endl;
    }
}

/**
         If there are multiple branches belonging to the same group
         set all the branches of the same group to their average
 */
void IQTreeMixHmm::setAvgLenEachBranchGrp() {
    size_t i,j;
    size_t treeIdx,branchIdx;
    double grp_len;
    int dim;
    
    // collect the branch lengths of the tree
    getAllBranchLengths();
    
    for (i = 0; i < branch_group.size(); i++) {
        grp_len = 0.0;
        dim = branch_group[i].size();
        for (j = 0; j < dim; j++) {
            treeIdx = branch_group[i].at(j) / nbranch;
            branchIdx = branch_group[i].at(j) % nbranch;
            if (allbranchlens[treeIdx].at(branchIdx) < params->min_branch_length)
                grp_len += params->min_branch_length;
            else
                grp_len += allbranchlens[treeIdx].at(branchIdx);
        }
        grp_len = grp_len / (double)dim;
        for (j = 0; j < dim; j++) {
            treeIdx = branch_group[i].at(j) / nbranch;
            branchIdx = branch_group[i].at(j) % nbranch;
            allbranchlens[treeIdx].at(branchIdx) = grp_len;
        }
    }
    
    // save the updated branch lengths of the tree
    setAllBranchLengths();
}

// update the ptn_freq array according to the marginal probabilities along each site for each tree
void IQTreeMixHmm::computeFreqArray(bool need_computeLike, int update_which_tree) {
    double* mar_prob;
    // get marginal probabilities along each site for each tree
    getMarginalProb(need_computeLike, update_which_tree);
    // #pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1)
    for (size_t i = 0; i < ntree; i++) {
        PhyloTree* tree = at(i);
        // reset the array ptn_freq
        memset(tree->ptn_freq, 0, sizeof(double)*nptn);
        mar_prob = marginal_prob + i;
        for (size_t j = 0; j < nsite; j++) {
            int ptn = aln->getPatternID(j);
            tree->ptn_freq[ptn] += mar_prob[0];
            mar_prob += ntree;
        }
    }
}

// get marginal probabilities along each site for each tree
void IQTreeMixHmm::getMarginalProb(bool need_computeLike, int update_which_tree) {
    if (need_computeLike) {
        computeLogLikelihoodSiteTree(update_which_tree);
    }
    computeBackLikeArray();
    computeFwdLikeArray();
    computeMarginalProb();
}