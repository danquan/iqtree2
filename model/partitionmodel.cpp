/***************************************************************************
 *   Copyright (C) 2009 by BUI Quang Minh   *
 *   minh.bui@univie.ac.at   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include "partitionmodel.h"
#include "alignment/superalignment.h"
#include "model/rategamma.h"
#include "model/modelmarkov.h"
#include "utils/MPIHelper.h"

PartitionModel::PartitionModel()
        : ModelFactory()
{
	linked_alpha = -1.0;
    opt_gamma_invar = false;
}

PartitionModel::PartitionModel(Params &params, PhyloSuperTree *tree, ModelsBlock *models_block)
        : ModelFactory()
{
	store_trans_matrix = params.store_trans_matrix;
	is_storing = false;
	joint_optimize = params.optimize_model_rate_joint;
	fused_mix_rate = false;
    linked_alpha = -1.0;
    opt_gamma_invar = false;

	// create dummy model
	model = new ModelSubst(tree->aln->num_states);
	site_rate = new RateHeterogeneity();
	site_rate->setTree(tree);

//    string model_name = params.model_name;
    PhyloSuperTree::iterator it;
    int part;
    if (params.link_alpha) {
        params.gamma_shape = fabs(params.gamma_shape);
        linked_alpha = params.gamma_shape;
    }
    double init_by_divmat = false;
    if (params.model_name_init && strcmp(params.model_name_init, "DIVMAT") == 0) {
        init_by_divmat = true;
        params.model_name_init = NULL;
    }
    for (it = tree->begin(), part = 0; it != tree->end(); it++, part++) {
        ASSERT(!((*it)->getModelFactory()));
        string model_name = (*it)->aln->model_name;
        if (model_name == "") // if empty, take model name from command option
        	model_name = params.model_name;
        (*it)->setModelFactory(new ModelFactory(params, model_name, (*it), models_block));
        (*it)->setModel((*it)->getModelFactory()->model);
        (*it)->setRate((*it)->getModelFactory()->site_rate);

        // link models between partitions
        if (params.link_model) {
            (*it)->getModel()->fixParameters(true);
            if (linked_models.find((*it)->getModel()->getName()) == linked_models.end()) {
                linked_models[(*it)->getModel()->getName()] = (*it)->getModel();
            }
        } else if ((*it)->aln->getNSeq() < tree->aln->getNSeq() && params.partition_type != TOPO_UNLINKED &&
            (*it)->getModel()->freq_type == FREQ_EMPIRICAL && (*it)->aln->seq_type != SEQ_CODON) {
        	// modify state_freq to account for empty sequences
        	(*it)->aln->computeStateFreq((*it)->getModel()->state_freq, (*it)->aln->getNSite() * (tree->aln->getNSeq() - (*it)->aln->getNSeq()));
        	(*it)->getModel()->decomposeRateMatrix();
        }
        
        //string taxa_set = ((SuperAlignment*)tree->aln)->getPattern(part);
        //(*it)->copyTree(tree, taxa_set);
        //(*it)->drawTree(cout);
    }
    if (init_by_divmat) {
        ASSERT(0 && "init_by_div_mat not working");
        int nstates = linked_models.begin()->second->num_states;
        double *pair_freq = new double[nstates * nstates];
        double *state_freq = new double[nstates];
        tree->aln->computeDivergenceMatrix(pair_freq, state_freq);
        /*
        MatrixXd divmat = Map<Matrix<double,Dynamic, Dynamic, RowMajor> > (pair_freq, nstates, nstates);
        cout << "DivMat: " << endl << divmat << endl;
        auto pi = Map<VectorXd>(state_freq, nstates);
        MatrixXd Q = (pi.asDiagonal() * divmat).log();
        cout << "Q: " << endl << Q << endl;
        cout << "rowsum: " << Q.rowwise().sum() << endl;
        Map<Matrix<double,Dynamic, Dynamic, RowMajor> >(pair_freq, nstates, nstates) = Q;
         */
        ((ModelMarkov*)linked_models.begin()->second)->setFullRateMatrix(pair_freq, state_freq);
        ((ModelMarkov*)linked_models.begin()->second)->decomposeRateMatrix();
        delete [] state_freq;
        delete [] pair_freq;

    } else
    for (auto mit = linked_models.begin(); mit != linked_models.end(); mit++) {
        PhyloSuperTree *stree = (PhyloSuperTree*)site_rate->phylo_tree;
        if (mit->second->freq_type != FREQ_ESTIMATE && mit->second->freq_type != FREQ_EMPIRICAL)
            continue;
        // count state occurrences
        size_t *sum_state_counts = NULL;
        int num_parts = 0;
        for (it = stree->begin(); it != stree->end(); it++) {
            if ((*it)->getModel()->getName() == mit->second->getName()) {
                num_parts++;
                if ((*it)->aln->seq_type == SEQ_CODON)
                    outError("Linking codon models not supported");
                if ((*it)->aln->seq_type == SEQ_POMO)
                    outError("Linking POMO models not supported");
                size_t state_counts[(*it)->aln->STATE_UNKNOWN+1];
                size_t unknown_states = 0;
                if( params.partition_type != TOPO_UNLINKED)
                    unknown_states = (*it)->aln->getNSite() * (tree->aln->getNSeq() - (*it)->aln->getNSeq());
                (*it)->aln->countStates(state_counts, unknown_states);
                if (!sum_state_counts) {
                    sum_state_counts = new size_t[(*it)->aln->STATE_UNKNOWN+1];
                    memset(sum_state_counts, 0, sizeof(size_t)*((*it)->aln->STATE_UNKNOWN+1));
                }
                for (int state = 0; state <= (*it)->aln->STATE_UNKNOWN; ++state) {
                    sum_state_counts[state] += state_counts[state];
                }
            }
        }
        cout << "Linking " << mit->first << " model across " << num_parts << " partitions" << endl;
        int nstates = mit->second->num_states;
        double sum_state_freq[nstates];
        // convert counts to frequencies
        for (it = stree->begin(); it != stree->end(); it++) {
            if ((*it)->getModel()->getName() == mit->second->getName()) {
                (*it)->aln->convertCountToFreq(sum_state_counts, sum_state_freq);
                break;
            }
        }

        cout << "Mean state frequencies:";
        int prec = cout.precision(8);
        for (int state = 0; state < mit->second->num_states; state++)
            cout << " " << sum_state_freq[state];
        cout << endl;
        cout.precision(prec);

        for (it = stree->begin(); it != stree->end(); it++)
            if ((*it)->getModel()->getName() == mit->second->getName()) {
                ((ModelMarkov*)(*it)->getModel())->adaptStateFrequency(sum_state_freq);
                (*it)->getModel()->decomposeRateMatrix();
            }
        delete [] sum_state_counts;
    }
}

void PartitionModel::setCheckpoint(Checkpoint *checkpoint) {
	ModelFactory::setCheckpoint(checkpoint);
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    for (PhyloSuperTree::iterator it = tree->begin(); it != tree->end(); it++)
		(*it)->getModelFactory()->setCheckpoint(checkpoint);
}

void PartitionModel::startCheckpoint() {
    checkpoint->startStruct("PartitionModel");
}

void PartitionModel::saveCheckpoint() {
    startCheckpoint();
    CKP_SAVE(linked_alpha);
    for (auto it = linked_models.begin(); it != linked_models.end(); it++) {
        checkpoint->startStruct(it->first);
        bool fixed = it->second->fixParameters(false);
        it->second->saveCheckpoint();
        it->second->fixParameters(fixed);
        checkpoint->endStruct();
    }
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    int part = 0;
    for (PhyloSuperTree::iterator it = tree->begin(); it != tree->end(); it++, part++) {
        checkpoint->startStruct((*it)->aln->name);
        (*it)->getModelFactory()->saveCheckpoint();
        checkpoint->endStruct();
    }
    endCheckpoint();

    CheckpointFactory::saveCheckpoint();
}

void PartitionModel::restoreCheckpoint() {
    CheckpointFactory::restoreCheckpoint();
    startCheckpoint();
    CKP_RESTORE(linked_alpha);

    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    int part = 0;
    for (PhyloSuperTree::iterator it = tree->begin(); it != tree->end(); it++, part++) {
        checkpoint->startStruct((*it)->aln->name);
        (*it)->getModelFactory()->restoreCheckpoint();
        checkpoint->endStruct();
    }

    // restore linked models
    for (auto it = linked_models.begin(); it != linked_models.end(); it++) {
        checkpoint->startStruct(it->first);
        for (auto tit = tree->begin(); tit != tree->end(); tit++)
            if ((*tit)->getModel()->getName() == it->first) {
                bool fixed = (*tit)->getModel()->fixParameters(false);
                (*tit)->getModel()->restoreCheckpoint();
                (*tit)->getModel()->fixParameters(fixed);
            }
        checkpoint->endStruct();
    }
    
    endCheckpoint();
}

bool PartitionModel::isReversible() {
    // check that all sub-models must be reversible
    PhyloSuperTree *super_tree = (PhyloSuperTree*)site_rate->getTree();
    for (auto tree : *super_tree) {
        if (!tree->getModelFactory()->isReversible())
            return false; // at least one sub-model is non-reversible
    }
    return true;
}

int PartitionModel::getNParameters(int brlen_type) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
	int df = 0;
    for (PhyloSuperTree::iterator it = tree->begin(); it != tree->end(); it++) {
    	df += (*it)->getModelFactory()->getNParameters(brlen_type);
    }
    if (linked_alpha > 0)
        df ++;
    for (auto it = linked_models.begin(); it != linked_models.end(); it++) {
        bool fixed = it->second->fixParameters(false);
        df += it->second->getNDim() + it->second->getNDimFreq();
        it->second->fixParameters(fixed);
    }
    return df;
}

double PartitionModel::computeFunction(double shape) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    double res = 0.0;
    int ntrees = tree->size();
    linked_alpha = shape;
    if (tree->part_order.empty()) tree->computePartitionOrder();
#ifdef _OPENMP
#pragma omp parallel for reduction(+: res) schedule(dynamic) if(tree->num_threads > 1)
#endif
    for (int j = 0; j < ntrees; j++) {
        int i = tree->part_order[j];
        if (tree->at(i)->getRate()->isGammaRate())
            res += tree->at(i)->getRate()->computeFunction(shape);
    }
    if (res == 0.0) {
        outError("No partition has Gamma rate heterogeneity!");
    }
	return res;
}

double PartitionModel::optimizeLinkedAlpha(bool write_info, double gradient_epsilon) {
    if (write_info) {
        cout << "Optimizing linked gamma shape..." << endl;
    }
	double negative_lh;
	double current_shape = linked_alpha;
	double ferror, optx;
	optx = minimizeOneDimen(site_rate->getTree()->params->min_gamma_shape, current_shape, MAX_GAMMA_SHAPE, max(gradient_epsilon, TOL_GAMMA_SHAPE), &negative_lh, &ferror);
    double tree_lh = site_rate->getTree()->computeLikelihood();
    if (write_info) {
        cout << "Linked alpha across partitions: " << linked_alpha << endl;
        cout << "Linked alpha log-likelihood: " << tree_lh << endl;
    }
	return tree_lh;
    
}

int PartitionModel::getNDim() {
    return model->getNDim();
}

double PartitionModel::targetFunk(double x[]) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    
    double res = 0;
    int ntrees = tree->size();
    if (tree->part_order.empty()) tree->computePartitionOrder();

    if (Params::getInstance().fpqmaker) {
        DoubleVector results(tree->size());        
        // clock_t start = clock();
        
        #ifdef _OPENMP
        #pragma omp parallel if (tree->num_threads > 1)
        #endif
        {
            while (true) {
                int i;
                #pragma omp critical
                i = MPIHelper::getInstance().getTask();
            
                if (i >= ntrees) {
                    break;
                }
                i = tree->part_order[i];
                ModelSubst *part_model = tree->at(i)->getModel();
                if (part_model->getName() != model->getName())
                    continue;
                bool fixed = part_model->fixParameters(false);
                results[i] = part_model->targetFunk(x);
                part_model->fixParameters(fixed);    
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (MPIHelper::getInstance().isMaster()) {
            MPIHelper::getInstance().setTask(- ntrees - MPIHelper::getInstance().getNumProcesses() * Params::getInstance().num_threads);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        #ifdef _IQTREE_MPI
            results = MPIHelper::getInstance().sumProcs(results);
        #endif
            for (auto e : results)
             res += e;
    } else if (Params::getInstance().pqmaker) {
        /*----------------------------------- Run pQMaker here ----------------------------------*/
        DoubleVector results(tree->size());

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+ : res) schedule(dynamic) if (tree->num_threads > 1)
        #endif
        #ifdef _IQTREE_MPI
            for (int j = 0; j < tree->procSize(); j++) {
                int i = tree->proc_part_order[j];
        #else
            for (int j = 0; j < tree->size(); j++) {
                int i = tree->part_order[j];
        #endif
                ModelSubst *part_model = tree->at(i)->getModel();
                if (part_model->getName() != model->getName())
                    continue;
                bool fixed = part_model->fixParameters(false);
                results[i] = part_model->targetFunk(x);
                part_model->fixParameters(fixed);
            }

        #ifdef _IQTREE_MPI
            results = MPIHelper::getInstance().sumProcs(results);
        #endif
            for (auto e : results)
             res += e;
    } else {
        /*----------------------------------- Run QMaker here ----------------------------------*/

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+: res) schedule(dynamic) if(tree->num_threads > 1)
        #endif
        for (int j = 0; j < ntrees; j++) {
            int i = tree->part_order[j];
            ModelSubst *part_model = tree->at(i)->getModel();
            if (part_model->getName() != model->getName())
                continue;
            bool fixed = part_model->fixParameters(false);
            res += part_model->targetFunk(x);
            part_model->fixParameters(fixed);
        }
    }
    if (res == 0.0)
        outError("No partition has model ", model->getName());
    return res;
}

void PartitionModel::setVariables(double *variables) {
    model->setVariables(variables);
}

bool PartitionModel::getVariables(double *variables) {
    bool changed = false;
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    for (auto it = tree->begin(); it != tree->end(); it++)
        if ((*it)->getModel()->getName() == model->getName())
            changed |= (*it)->getModel()->getVariables(variables);
    return changed;
}

void PartitionModel::scaleStateFreq(bool sum_one) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    for (auto it = tree->begin(); it != tree->end(); it++)
        if ((*it)->getModel()->getName() == model->getName())
            ((ModelMarkov*)(*it)->getModel())->scaleStateFreq(sum_one);
}

double PartitionModel::optimizeLinkedModel(bool write_info, double gradient_epsilon) {
    int ndim = getNDim();
    
    // return if nothing to be optimized
    if (ndim == 0) return 0.0;
    
    if (write_info)
        cout << "Optimizing linked " << model->getName() << " parameters across all partitions (" << ndim << " free parameters)" << endl;
    
    if (verbose_mode >= VB_MAX)
        cout << "Optimizing " << model->name << " model parameters..." << endl;
    
    //if (freq_type == FREQ_ESTIMATE) scaleStateFreq(false);
    
    double *variables = new double[ndim+1]; // used for BFGS numerical recipes
    double *variables2 = new double[ndim+1]; // used for L-BFGS-B
    double *upper_bound = new double[ndim+1];
    double *lower_bound = new double[ndim+1];
    bool *bound_check = new bool[ndim+1];
    double score;
    
    
    // by BFGS algorithm
    setVariables(variables);
    setVariables(variables2);
    ((ModelMarkov*)model)->setBounds(lower_bound, upper_bound, bound_check);
    // expand the bound for linked model
//    for (int i = 1; i <= ndim; i++) {
//        lower_bound[i] = MIN_RATE*0.2;
//        upper_bound[i] = MAX_RATE*2.0;
//    }

//    if (Params::getInstance().optimize_alg.find("BFGS-B") == string::npos)
        score = -minimizeMultiDimen(variables, ndim, lower_bound, upper_bound, bound_check, max(gradient_epsilon, TOL_RATE));
//    else
//        score = -L_BFGS_B(ndim, variables+1, lower_bound+1, upper_bound+1, max(gradient_epsilon, TOL_RATE));

    bool changed = getVariables(variables);

    /* 2019-09-05: REMOVED due to numerical issue (NAN) with L-BFGS-B
    // 2017-12-06: more robust optimization using 2 different routines
    // when estimates are at boundary
    score = -minimizeMultiDimen(variables, ndim, lower_bound, upper_bound, bound_check, max(gradient_epsilon, TOL_RATE));
    bool changed = getVariables(variables);
    
    if (model->isUnstableParameters()) {
        // parameters at boundary, restart with L-BFGS-B with parameters2
        double score2 = -L_BFGS_B(ndim, variables2+1, lower_bound+1, upper_bound+1, max(gradient_epsilon, TOL_RATE));
        if (score2 > score+0.1) {
            if (verbose_mode >= VB_MED)
                cout << "NICE: L-BFGS-B found better parameters with LnL=" << score2 << " than BFGS LnL=" << score << endl;
            changed = getVariables(variables2);
            score = score2;
        } else {
            // otherwise, revert what BFGS found
            changed = getVariables(variables);
        }
    }
    */
    
    // BQM 2015-09-07: normalize state_freq
    if (model->isReversible() && model->freq_type == FREQ_ESTIMATE) {
        scaleStateFreq(true);
        changed = true;
    }
    if (changed) {
        PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
        for (auto it = tree->begin(); it != tree->end(); it++)
            if ((*it)->getModel()->getName() == model->getName())
                (*it)->getModel()->decomposeRateMatrix();
        site_rate->phylo_tree->clearAllPartialLH();
        score = site_rate->phylo_tree->computeLikelihood();
    }
    
    delete [] bound_check;
    delete [] lower_bound;
    delete [] upper_bound;
    delete [] variables2;
    delete [] variables;
    
    if (write_info) {
        cout << "Linked-model log-likelihood: " << score << endl;
    }

    return score;
}

double PartitionModel::optimizeLinkedModels(bool write_info, double gradient_epsilon) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    double tree_lh;
    for (auto it = linked_models.begin(); it != linked_models.end(); it++) {
        ModelSubst *saved_model = model;
        model = it->second;
        PhyloSuperTree::iterator part_tree;
        // un-fix model parameters
        for (part_tree = tree->begin(); part_tree != tree->end(); part_tree++)
            if ((*part_tree)->getModel()->getName() == model->getName())
                (*part_tree)->getModel()->fixParameters(false);
        
        // main call to optimize linked model parameters
        tree_lh = optimizeLinkedModel(write_info, gradient_epsilon);
        
        // fix model parameters again
        for (part_tree = tree->begin(); part_tree != tree->end(); part_tree++)
            if ((*part_tree)->getModel()->getName() == model->getName())
                (*part_tree)->getModel()->fixParameters(true);
        
        saveCheckpoint();
        getCheckpoint()->dump();
        model = saved_model;
    }

    return site_rate->phylo_tree->computeLikelihood();
}

void PartitionModel::reportLinkedModel(ostream &out) {
    if (linked_alpha > 0.0)
        out << "Linked alpha across partitions: " << linked_alpha << endl;
    for (auto it = linked_models.begin(); it != linked_models.end(); it++) {
        out << "Linked model " << it->first << ": " << endl;
        it->second->report(out);
    }
}

bool PartitionModel::isLinkedModel() {
    return Params::getInstance().link_alpha || (linked_models.size()>0);
}

double PartitionModel::optimizeParameters(int fixed_len, bool write_info, double logl_epsilon, double gradient_epsilon) {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();
    double prev_tree_lh = -DBL_MAX, tree_lh = 0.0;
    int ntrees = tree->size();
    DoubleVector tree_lhs(ntrees, 0.0);

    for (int step = 0; step < Params::getInstance().model_opt_steps; step++) {
        tree_lh = 0.0;
        if (Params::getInstance().pqmaker || Params::getInstance().fpqmaker) tree_lhs = DoubleVector(ntrees, 0.0);
        if (tree->part_order.empty()) tree->computePartitionOrder();

        if (false && Params::getInstance().fpqmaker) {
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+: tree_lh) schedule(dynamic) if(tree->num_threads > 1)
    #endif
            for (int j = 0; j < ntrees; ++j) {
                int part;
                #pragma omp critical
                {
                    part = MPIHelper::getInstance().getTask();
                }

                if (part >= ntrees) continue;
                tree->proc_part_order.push_back(part);
                printf("Process %d: Partition %d\n", MPIHelper::getInstance().getProcessID(), part);
                
                double score;
                
                if (opt_gamma_invar)
                    score = tree->at(part)->getModelFactory()->optimizeParametersGammaInvar(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                else
                    score = tree->at(part)->getModelFactory()->optimizeParameters(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                tree_lhs[part] = score;
                //tree_lh += score;

                // Canh: quick fix
                // Currently only alignments processed by master in MPI version are logged
                // Remove logging to avoid misunderstanding
                // #ifndef _IQTREE_MPI
                //     if (write_info)
                //     #ifdef _OPENMP
                //     #pragma omp critical
                //     #endif
                //     {
                //         cout << "Partition " << tree->at(part)->aln->name
                //             << " / Model: " << tree->at(part)->getModelName()
                //             << " / df: " << tree->at(part)->getModelFactory()->getNParameters(fixed_len)
                //             << " / LogL: " << score << endl;
                //     }
                // #endif // _IQTREE_MPI
            }

            MPI_Barrier(MPI_COMM_WORLD);
            
            if (MPIHelper::getInstance().isMaster()) {
                MPIHelper::getInstance().setTask(- ntrees * MPIHelper::getInstance().getNumProcesses());
            }

            MPI_Barrier(MPI_COMM_WORLD);

        //return ModelFactory::optimizeParameters(fixed_len, write_info);
        #ifdef _IQTREE_MPI
            tree_lhs = MPIHelper::getInstance().sumProcs(tree_lhs);
            syncBranchLengths();
            tree->proc_part_order.clear();
        #endif
            for (auto e: tree_lhs)
                tree_lh += e;
        } else 
        if (Params::getInstance().pqmaker) {
            /*----------------------------------- Run pQMaker here ----------------------------------*/
            #ifdef _IQTREE_MPI
            int proc_ntrees = tree->procSize();
            #endif

            #ifdef _OPENMP
            #pragma omp parallel for reduction(+: tree_lh) schedule(dynamic) if(tree->num_threads > 1)
            #endif
        #ifdef _IQTREE_MPI
            for (int i = 0; i < proc_ntrees; i++) {
                int part = tree->proc_part_order[i];
        #else
            for (int i = 0; i < ntrees; i++) {
                int part = tree->part_order[i];
        #endif
                double score;
                
                if (opt_gamma_invar)
                    score = tree->at(part)->getModelFactory()->optimizeParametersGammaInvar(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                else
                    score = tree->at(part)->getModelFactory()->optimizeParameters(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                tree_lhs[part] = score;
                //tree_lh += score;

                // Canh: quick fix
                // Currently only alignments processed by master in MPI version are logged
                // Remove logging to avoid misunderstanding
            #ifndef _IQTREE_MPI
                if (write_info)
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    cout << "Partition " << tree->at(part)->aln->name
                        << " / Model: " << tree->at(part)->getModelName()
                        << " / df: " << tree->at(part)->getModelFactory()->getNParameters(fixed_len)
                        << " / LogL: " << score << endl;
                }
            #endif // _IQTREE_MPI
        }

        //return ModelFactory::optimizeParameters(fixed_len, write_info);
        #ifdef _IQTREE_MPI
            tree_lhs = MPIHelper::getInstance().sumProcs(tree_lhs);
            syncBranchLengths();
        #endif
            for (auto e: tree_lhs)
                tree_lh += e;

        } else {
            /*----------------------------------- Run QMaker here ----------------------------------*/
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+: tree_lh) schedule(dynamic) if(tree->num_threads > 1)
            #endif
            for (int i = 0; i < ntrees; i++) {
                int part = tree->part_order[i];
                double score;
                if (opt_gamma_invar)
                    score = tree->at(part)->getModelFactory()->optimizeParametersGammaInvar(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                else
                    score = tree->at(part)->getModelFactory()->optimizeParameters(fixed_len,
                        write_info && verbose_mode >= VB_MED,
                        logl_epsilon/min(ntrees,10), gradient_epsilon/min(ntrees,10));
                tree_lh += score;
                if (write_info)
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    cout << "Partition " << tree->at(part)->aln->name
                        << " / Model: " << tree->at(part)->getModelName()
                        << " / df: " << tree->at(part)->getModelFactory()->getNParameters(fixed_len)
                    << " / LogL: " << score << endl;
                }
            }
            //return ModelFactory::optimizeParameters(fixed_len, write_info);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (!isLinkedModel())
            break;

        if (verbose_mode >= VB_MED || write_info)
            cout << step+1 << ". Log-likelihood: " << tree_lh << endl;

        if (tree->params->link_alpha) {
            tree_lh = optimizeLinkedAlpha(write_info, gradient_epsilon);
        }

        // optimize linked models
        if (!linked_models.empty()) {
            double new_tree_lh = optimizeLinkedModels(write_info, gradient_epsilon);
            ASSERT(new_tree_lh > tree_lh - 0.1);
            tree_lh = new_tree_lh;
        }
        
        if (tree_lh-logl_epsilon*10 < prev_tree_lh)
            break;
        prev_tree_lh = tree_lh;
    }
    
    if (verbose_mode >= VB_MED || write_info)
		cout << "Optimal log-likelihood: " << tree_lh << endl;
    // write linked_models
    if (verbose_mode <= VB_MIN && write_info) {
        for (auto it = linked_models.begin(); it != linked_models.end(); it++)
            it->second->writeInfo(cout);
    }
    return tree_lh;
}


double PartitionModel::optimizeParametersGammaInvar(int fixed_len, bool write_info, double logl_epsilon, double gradient_epsilon) {
    opt_gamma_invar = true;
    double tree_lh = optimizeParameters(fixed_len, write_info, logl_epsilon, gradient_epsilon);
    opt_gamma_invar = false;
    return tree_lh;
}


PartitionModel::~PartitionModel()
{
}

bool PartitionModel::isUnstableParameters() {
    PhyloSuperTree *tree = (PhyloSuperTree*)site_rate->getTree();

	for (PhyloSuperTree::iterator it = tree->begin(); it != tree->end(); it++)
		if ((*it)->getModelFactory()->isUnstableParameters()) {
			return true;
		}
	return false;
}

#ifdef _IQTREE_MPI
void PartitionModel::syncBranchLengths()
{
    int nprocs = MPIHelper::getInstance().getNumProcesses();
    if (nprocs == 1)
        return;

    PhyloSuperTree *tree = (PhyloSuperTree *)site_rate->getTree();
    int ntrees = tree->size();
    int proc_ntrees = tree->procSize();
    vector<DoubleVector> proc_blens(ntrees);

    for (int i = 0; i < proc_ntrees; i++)
    {
        int part = tree->proc_part_order[i];
        tree->at(part)->saveBranchLengths(proc_blens[part]);
    }

    vector<DoubleVector> all_blens = MPIHelper::getInstance().gatherAllVectors(proc_blens);

    for (int i = 0; i < ntrees; i++)
        for (int j = 0; j < nprocs; j++)
            if (!all_blens[ntrees * j + i].empty() && j != MPIHelper::getInstance().getProcessID())
                tree->at(i)->restoreBranchLengths(all_blens[ntrees * j + i]);
}
#endif
