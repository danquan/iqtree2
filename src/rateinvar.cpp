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
#include "rateinvar.h"

RateInvar::RateInvar(PhyloTree *tree)
 : RateHeterogeneity()
{
	p_invar = 0.0;
	phylo_tree = tree;
	name = "+I";
}

double RateInvar::computeFunction(double p_invar_value) {
	p_invar = p_invar_value;
	//phylo_tree->clearAllPartialLh();
	return -phylo_tree->computeLikelihood();
}

double RateInvar::optimizeParameters() {
	if (verbose_mode >= VB_MAX)
		cout << "Optimizing proportion of invariable sites..." << endl;
	double negative_lh;
	double ferror;
	p_invar = minimizeOneDimen(1e-6, p_invar, phylo_tree->aln->frac_const_sites, 1e-6, &negative_lh, &ferror);
	//phylo_tree->clearAllPartialLh();
	return -negative_lh;
}

void RateInvar::writeInfo(ostream &out) {
	out << "Proportion of invariable sites: " << p_invar << endl;
}

void RateInvar::writeParameters(ostream &out) {
	out << "\t" << p_invar;
}

