//
// Created by tung on 6/18/15.
//

#include "MPIHelper.h"
#include "timeutil.h"

/**
 *  Initialize the single getInstance of MPIHelper
 */

MPIHelper& MPIHelper::getInstance() {
    static MPIHelper instance;
#ifndef _IQTREE_MPI
    instance.setProcessID(0);
    instance.setNumProcesses(1);
#endif
    return instance;
}

void MPIHelper::init(int argc, char *argv[]) {
#ifdef _IQTREE_MPI
    int n_tasks, task_id;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        outError("MPI initialization failed!");
    }
    MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    setNumProcesses(n_tasks);
    setProcessID(task_id);
    setNumTreeReceived(0);
    setNumTreeSent(0);
    setNumNNISearch(0);
#endif
}

void MPIHelper::initSharedMemory() {
#ifdef _IQTREE_MPI
    if (Params::getInstance().fpqmaker) {
        if (getNumProcesses() > 1) {
            MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_counter, &shmwin);
            if (isMaster()) {
                shared_counter[0] = 0;
            }
        } else {
            shared_counter = new int[1];
            shared_counter[0] = 0;
        }
    }
    
    if (Params::getInstance().split) {
        if (getNumProcesses() > 1) {
            MPI_Win_allocate(sizeof(int) * 3, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_counter, &shmwin);
            if (isMaster()) {
                shared_counter[0] = 0;
                shared_counter[1] = 0;
                shared_counter[2] = 0;
            }
        } else {
            shared_counter = new int[3];
            shared_counter[0] = 0;
            shared_counter[1] = 0;
            shared_counter[2] = 0;
        }
    }
    barrier();
#endif
}

int MPIHelper::increment(int id, bool enableLock) {
#ifdef _IQTREE_MPI
    if (getNumProcesses() > 1) {
        int one = 1, ret;
        if (enableLock) MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shmwin);
        MPI_Fetch_and_op(&one, &ret, MPI_INT, 0, id, MPI_SUM, shmwin);
        if (enableLock) MPI_Win_unlock(0, shmwin);
        return ret;
    } else {
        return shared_counter[id]++;
    }
#endif
}

int MPIHelper::decrement(int id, bool enableLock) {
#ifdef _IQTREE_MPI
    if (getNumProcesses() > 1) {
        int minus_one = -1, ret;
        if (enableLock) MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shmwin);
        MPI_Fetch_and_op(&minus_one, &ret, MPI_INT, 0, id, MPI_SUM, shmwin);
        if (enableLock) MPI_Win_unlock(0, shmwin);
        return ret;
    } else {
        return shared_counter[id]--;
    }
#else
    assert(0);
#endif
}

void MPIHelper::lock() {
#ifdef _IQTREE_MPI
    if (getNumProcesses() > 1) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shmwin);
    }
#endif
}


void MPIHelper::unlock() {
#ifdef _IQTREE_MPI
    if (getNumProcesses() > 1) {
        MPI_Win_unlock(0, shmwin);
    }
#endif
}

void MPIHelper::setTask(int delta) {
#ifdef _IQTREE_MPI
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shmwin);
    MPI_Accumulate(&delta, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, shmwin);
    MPI_Win_unlock(0, shmwin);
#else
    assert(0);
#endif
}

int MPIHelper::getSharedCounter(int id, bool enableLock) {
#ifdef _IQTREE_MPI
    if (getNumProcesses() > 1) {
        int ret;
        if (enableLock) MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shmwin);
        MPI_Get(&ret, 1, MPI_INT, 0, id, 1, MPI_INT, shmwin);
        if (enableLock) MPI_Win_unlock(0, shmwin);
        return ret;
    } else return shared_counter[id];
#endif
}

void MPIHelper::finalize() {
#ifdef _IQTREE_MPI
    if (Params::getInstance().fpqmaker || Params::getInstance().split)
        MPI_Win_free(&shmwin);
    MPI_Finalize();
#endif
}

void MPIHelper::syncRandomSeed() {
#ifdef _IQTREE_MPI
    unsigned int rndSeed;
    if (MPIHelper::getInstance().isMaster()) {
        rndSeed = Params::getInstance().ran_seed;
    }
    // Broadcast random seed
    MPI_Bcast(&rndSeed, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
    if (MPIHelper::getInstance().isWorker()) {
        //        Params::getInstance().ran_seed = rndSeed + task_id * 100000;
        Params::getInstance().ran_seed = rndSeed;
        //        printf("Process %d: random_seed = %d\n", task_id, Params::getInstance().ran_seed);
    }
#endif
}

int MPIHelper::countSameHost() {
#ifdef _IQTREE_MPI
    // detect if processes are in the same host
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int resultlen;
    /*int pID =*/ (void) MPIHelper::getInstance().getProcessID();
    MPI_Get_processor_name(host_name, &resultlen);
    char *host_names;
    host_names = new char[MPI_MAX_PROCESSOR_NAME * MPIHelper::getInstance().getNumProcesses()];
    
    MPI_Allgather(host_name, resultlen+1, MPI_CHAR, host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               MPI_COMM_WORLD);
    
    int count = 0;
    for (int i = 0; i < MPIHelper::getInstance().getNumProcesses(); i++)
        if (strcmp(&host_names[i*MPI_MAX_PROCESSOR_NAME], host_name) == 0)
            count++;
    delete [] host_names;
    if (count>1)
        cout << "NOTE: " << count << " processes are running on the same host " << host_name << endl; 
    return count;
#else
    return 1;
#endif
}

bool MPIHelper::gotMessage() {
    // Check for incoming messages
    if (getNumProcesses() == 1)
        return false;
#ifdef _IQTREE_MPI
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag)
        return true;
    else
        return false;
#else
    return false;
#endif
}



#ifdef _IQTREE_MPI
void MPIHelper::sendString(string &str, int dest, int tag) {
    char *buf = (char*)str.c_str();
    MPI_Send(buf, str.length()+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
}

void MPIHelper::sendCheckpoint(Checkpoint *ckp, int dest) {
    stringstream ss;
    ckp->dump(ss);
    string str = ss.str();
    sendString(str, dest, TREE_TAG);
}


int MPIHelper::recvString(string &str, int src, int tag) {
    MPI_Status status;
    MPI_Probe(src, tag, MPI_COMM_WORLD, &status);
    int msgCount;
    MPI_Get_count(&status, MPI_CHAR, &msgCount);
    // receive the message
    char *recvBuffer = new char[msgCount];
    MPI_Recv(recvBuffer, msgCount, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
    str = recvBuffer;
    delete [] recvBuffer;
    return status.MPI_SOURCE;
}

int MPIHelper::recvCheckpoint(Checkpoint *ckp, int src) {
    string str;
    int proc = recvString(str, src, TREE_TAG);
    stringstream ss(str);
    ckp->load(ss);
    return proc;
}

void MPIHelper::broadcastCheckpoint(Checkpoint *ckp) {
    int msgCount = 0;
    stringstream ss;
    string str;
    if (isMaster()) {
        ckp->dump(ss);
        str = ss.str();
        msgCount = str.length()+1;
    }

    // broadcast the count for workers
    MPI_Bcast(&msgCount, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);

    char *recvBuffer = new char[msgCount];
    if (isMaster())
        memcpy(recvBuffer, str.c_str(), msgCount);

    // broadcast trees to workers
    MPI_Bcast(recvBuffer, msgCount, MPI_CHAR, PROC_MASTER, MPI_COMM_WORLD);

    if (isWorker()) {
        ss.clear();
        ss.str(recvBuffer);
        ckp->load(ss);
    }
    delete [] recvBuffer;
}

void MPIHelper::gatherCheckpoint(Checkpoint *ckp) {
    stringstream ss;
    ckp->dump(ss);
    string str = ss.str();
    int msgCount = str.length();

    // first send the counts to MASTER
    int *msgCounts = NULL, *displ = NULL;
    char *recvBuffer = NULL;
    int totalCount = 0;

    if (isMaster()) {
        msgCounts = new int[getNumProcesses()];
        displ = new int[getNumProcesses()];
    }
    MPI_Gather(&msgCount, 1, MPI_INT, msgCounts, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);

    // now real contents to MASTER
    if (isMaster()) {
        for (int i = 0; i < getNumProcesses(); i++) {
            displ[i] = totalCount;
            totalCount += msgCounts[i];
        }
        recvBuffer = new char[totalCount+1];
        memset(recvBuffer, 0, totalCount+1);
    }
    char *buf = (char*)str.c_str();
    MPI_Gatherv(buf, msgCount, MPI_CHAR, recvBuffer, msgCounts, displ, MPI_CHAR, PROC_MASTER, MPI_COMM_WORLD);

    if (isMaster()) {
        // now decode the buffer
        ss.clear();
        ss.str(recvBuffer);
        ckp->load(ss);

        delete [] recvBuffer;
        delete [] displ;
        delete [] msgCounts;
    }
}

#endif

MPIHelper::~MPIHelper() {
//    cleanUpMessages();
}

#ifdef _IQTREE_MPI
DoubleVector MPIHelper::sumProcs(DoubleVector vals)
{
    int proc_size = vals.size();
    DoubleVector sum_vals(proc_size);
    MPI_Allreduce(vals.data(), sum_vals.data(), proc_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum_vals;
}

IntVector MPIHelper::getProcVector(const vector<IntVector> &vts)
{
    IntVector cnt_vt, offset_vt, flatten_vt;

    if (isMaster())
    {
        int offset = 0;
        for (const auto &vt : vts)
        {
            cnt_vt.push_back(vt.size());
            offset_vt.push_back(offset);
            offset += vt.size();
            flatten_vt.insert(flatten_vt.end(), vt.begin(), vt.end());
        }
    }

    int out_cnt;
    // Send cnt to each process
    MPI_Scatter(cnt_vt.data(), 1, MPI_INT, &out_cnt,
                1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);

    IntVector out_vt(out_cnt);
    // Send real contents to each process
    MPI_Scatterv(flatten_vt.data(), cnt_vt.data(), offset_vt.data(), MPI_INT,
                 out_vt.data(), out_cnt, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
    return out_vt;
}

vector<DoubleVector> MPIHelper::gatherAllVectors(const vector<DoubleVector> &vts)
{
    int flatten_sz = vts.size();
    for (const auto &vt : vts)
    {
        flatten_sz += vt.size();
    }
    DoubleVector flatten_vt(flatten_sz);

    int cnt = 0;
    for (const auto &vt : vts)
    {
        int vt_sz = vt.size();
        flatten_vt[cnt++] = vt_sz;
        memcpy(flatten_vt.data() + cnt, vt.data(), vt_sz * sizeof(double));
        cnt += vt_sz;
    }

    int nprocs = getNumProcesses();
    IntVector recv_cnts(nprocs);
    IntVector displ(nprocs);

    MPI_Allgather(&flatten_sz, 1, MPI_INT, recv_cnts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int recv_sz = 0;
    for (int i = 0; i < nprocs; i++)
    {
        displ[i] = recv_sz;
        recv_sz += recv_cnts[i];
    }

    DoubleVector recv_vt(recv_sz);
    MPI_Allgatherv(flatten_vt.data(), flatten_sz, MPI_DOUBLE, recv_vt.data(), recv_cnts.data(), displ.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    vector<DoubleVector> res_vts(vts.size() * nprocs);
    cnt = 0;
    for (int i = 0; i < res_vts.size(); i++)
    {
        int len = recv_vt[cnt++];
        res_vts[i] = DoubleVector(len);
        memcpy(res_vts[i].data(), recv_vt.data() + cnt, len * sizeof(double));
        cnt += len;
    }

    return res_vts;
}

#endif