# Copyright (C) 2009 Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from model import *
import dufvax
import pymc as pm
from generic_mbg import FieldStepper
import numpy as np
import os
root = os.path.split(dufvax.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)


f_labels = []
fs_have_nugget = {}
nugget_labels = {}
M_labels = {}
C_labels = {}
x_labels = {}
diags_safe = {}

def map_postproc():
    raise NotImplementedError

def validate_postproc(**non_cov_columns):
    raise NotImplementedError

metadata_keys = []

def mcmc_init(M):
    raise NotIMplementedError

non_cov_columns = {}