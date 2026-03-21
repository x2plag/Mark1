/***************************************************************************************************
 * Pollard–Kangaroo  (wrap-aware, user-configurable k, live counter, loop detector, restart counter)
 * Coded by DooKoo2
 * Load/Save DP tech by NoMachine
 * SSD rework (strong DP on SSD instead of RAM. Bloom – RAM, DP_table – SSD)
 * Added a few improvements with runtime security
 * Patched for higher performance (dual-hash DP table, better Bloom/DP sizing, batched I/O)
 *
 *  g++ Mark1.cpp Int.cpp SECP256K1.cpp Point.cpp Random.cpp IntMod.cpp IntGroup.cpp Timer.cpp \
 *      -O3 -march=native -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
 *      -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -fipa-modref -flto -fassociative-math \
 *      -fopenmp -mavx2 -mbmi2 -madx -std=c++17 -pthread -o Mark1
 *
 ***************************************************************************************************/
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#include <omp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Int.h"
#include "Point.h"
#include "SECP256K1.h"
#include "IntGroup.h"
#include "simd_block_bloom.h"

// ─── Misc ─────────────────────────────────────────────────────────────────────
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBULL;
    return x ^ (x>>31);
}
static std::string humanBytes(size_t bytes){
    constexpr const char* u[]{"B","Kb","Mb","Gb","Tb"};
    double v = double(bytes); int s = 0;
    while(v >= 1024.0 && s < 4){ v /= 1024.0; ++s; }
    std::ostringstream o;
    o << std::fixed << std::setprecision((v<10)?2:(v<100)?1:0) << v << u[s];
    return o.str();
}
static inline void intCopy(Int& d,const Int& s){ d.Set(&const_cast<Int&>(s)); }
static inline uint64_t IntLow64(const Int& n){ return n.bits64[0]; }
static inline int bitlen(const Int& n){ return const_cast<Int&>(n).GetBitLength(); }
static inline bool intGE(const Int& a,const Int& b){
    return const_cast<Int&>(a).IsGreaterOrEqual(&const_cast<Int&>(b));
}

// ─── Scalar256 ────────────────────────────────────────────────────────────────
struct Scalar256{ uint64_t w[4]; };
static inline void intToScalar(const Int& n, Scalar256& s){
    s.w[0]=n.bits64[0]; s.w[1]=n.bits64[1];
    s.w[2]=n.bits64[2]; s.w[3]=n.bits64[3];
}
static inline void scalarToInt(const Scalar256& s, Int& n){
    n.SetInt32(0);
    for(int i=3;i>=0;--i){ n.ShiftL(64); Int tmp(s.w[i]); n.Add(&tmp); }
}

using fp_t = uint64_t;

// ─── SSD DP storage ────────────────────────────────────────────────────────────
#pragma pack(push,1)
struct DPSlot{ fp_t fp; Scalar256 key; };
#pragma pack(pop)
static_assert(sizeof(DPSlot)==40);

struct DPStorage{
    size_t                        cap=0, mapBytes=0;
    int                           fd=-1;
    DPSlot*                       slots=nullptr;
    std::unique_ptr<std::atomic<uint8_t>[]> st_used, st_lock;
    std::atomic<size_t>           size{0};           
    std::atomic<size_t>           dirty{0};
    std::atomic<size_t>           flush_counter{0};   
    std::atomic<bool>             enable_flush{true};

    void   init(const std::string& path,size_t c);
    void   flushIfNeeded(size_t slotIdx) noexcept;
    void   fullSync() noexcept;
    void   close();
};
static DPStorage dp;

static constexpr size_t FLUSH_STEP = 1ull<<24;   

// ─── DPStorage impl ───────────────────────────────────────────────────────────
void DPStorage::init(const std::string& path,size_t c){
    cap=c; mapBytes=cap*sizeof(DPSlot);

    int flags = O_RDWR | O_CREAT;
#ifdef O_DIRECT
    flags |= O_DIRECT;         
#endif
#ifdef O_SYNC
    flags |= O_SYNC;           
#endif
    fd = ::open(path.c_str(),flags,0644);
    if(fd<0){
        perror("open(dp)"); throw std::runtime_error("open(dp)");
    }
    if(posix_fallocate(fd,0,mapBytes)){
        perror("fallocate(dp)"); throw std::runtime_error("fallocate(dp)");
    }
    void* p = mmap(nullptr,mapBytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(p==MAP_FAILED){
        perror("mmap(dp)"); throw std::runtime_error("mmap(dp)");
    }
    slots = static_cast<DPSlot*>(p);

    st_used = std::make_unique<std::atomic<uint8_t>[]>(cap);
    st_lock = std::make_unique<std::atomic<uint8_t>[]>(cap);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<cap;++i){
        st_used[i].store(0,std::memory_order_relaxed);
        st_lock[i].store(0,std::memory_order_relaxed);
    }
    madvise(slots,mapBytes,MADV_RANDOM);
}

void DPStorage::flushIfNeeded(size_t slotIdx) noexcept{
    if(!enable_flush.load(std::memory_order_relaxed)) return;
    if(flush_counter.fetch_add(1,std::memory_order_relaxed) % FLUSH_STEP == 0){
        size_t start = (slotIdx / FLUSH_STEP) * FLUSH_STEP;
        size_t end   = std::min(start + FLUSH_STEP, cap);
        size_t len   = (end - start) * sizeof(DPSlot);
        msync(reinterpret_cast<char*>(slots) + start*sizeof(DPSlot), len, MS_ASYNC);
    }
}
void DPStorage::fullSync() noexcept{
    msync(slots,mapBytes,MS_SYNC);
}
void DPStorage::close(){
    if(slots) munmap(slots,mapBytes);
    if(fd>=0) ::close(fd);
}

// ─── Curve ────────────────────────────────────────────────────────────────────
static const char *P_HEX="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
static const char *N_HEX="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
static Int P_PRIME, ORDER_N; static Secp256K1 secp;

static inline Point addP(const Point& a,const Point& b){
    if(const_cast<Int&>(a.x).IsZero() && const_cast<Int&>(a.y).IsZero()) return b;
    if(const_cast<Int&>(b.x).IsZero() && const_cast<Int&>(b.y).IsZero()) return a;
    return secp.AddDirect(const_cast<Point&>(a),const_cast<Point&>(b));
}
static inline Point mulP(const Int& k){ Int t(k); return secp.ComputePublicKey(&t); }

// ─── Bloom + DP API ───────────────────────────────────────────────────────────
static simd_bloom::SimdBlockFilterFixed<>* bloom=nullptr;
static std::atomic<uint64_t> dpDone{0};

inline bool sameScalar(const Scalar256& a,const Scalar256& b){
    return std::memcmp(&a,&b,sizeof(Scalar256))==0;
}

// ─── Dual-hash DP ────────────────────────────────────────────────────────────
bool dp_insert_unique(fp_t fp,const Int& idx){
    Int t(idx); t.Mod(&ORDER_N);
    Scalar256 key; intToScalar(t,key);

    size_t mask = dp.cap - 1;
    size_t h1   = fp & mask;
    size_t h2   = ((fp << 1) | 1) & mask;
    if(h2 == 0) h2 = 1;

    size_t h = h1;
    for(size_t i=0;i<dp.cap;++i){
        if(!dp.st_used[h].load(std::memory_order_acquire)){
            uint8_t exp=0;
            if(dp.st_lock[h].compare_exchange_strong(exp,1,std::memory_order_acq_rel)){
                if(!dp.st_used[h].load(std::memory_order_acquire)){
                    dp.slots[h].fp  = fp;
                    dp.slots[h].key = key;
                    dp.st_used[h].store(1,std::memory_order_release);
                    dp.st_lock[h].store(0,std::memory_order_release);

                    dp.size.fetch_add(1,std::memory_order_relaxed);
                    dp.flushIfNeeded(h);
                    dpDone.fetch_add(1,std::memory_order_relaxed);
                    return true;
                }
                dp.st_lock[h].store(0,std::memory_order_release);
            }
        }else if(dp.slots[h].fp==fp && sameScalar(dp.slots[h].key,key)){
            return false;
        }
        h = (h + h2) & mask;
    }
    return false;            
}
bool dp_find(fp_t fp,Int& out){
    size_t mask = dp.cap - 1;
    size_t h1   = fp & mask;
    size_t h2   = ((fp << 1) | 1) & mask;
    if(h2 == 0) h2 = 1;

    size_t h = h1;
    for(size_t i=0;i<dp.cap;++i){
        if(!dp.st_used[h].load(std::memory_order_acquire))
            return false;
        if(dp.slots[h].fp == fp){
            scalarToInt(dp.slots[h].key,out);
            return true;
        }
        h = (h + h2) & mask;
    }
    return false;
}

// ─── Binary DP I/O ────────────────────────────────────────────────────────────
#pragma pack(push,1)
struct DpItem{ fp_t fp; uint8_t priv[32]; };
#pragma pack(pop)

void saveDPBinary(const std::string& fn){
    std::ofstream f(fn,std::ios::binary|std::ios::trunc);
    if(!f){ std::cerr<<"[ERR] open "<<fn<<"\n"; return; }
    uint64_t cnt=0;
    for(size_t h=0;h<dp.cap;++h){
        if(!dp.st_used[h].load(std::memory_order_acquire)) continue;
        DpItem it{dp.slots[h].fp};
        Int p; scalarToInt(dp.slots[h].key,p); p.Get32Bytes(it.priv);
        f.write(reinterpret_cast<char*>(&it),sizeof(it)); ++cnt;
    }
    std::cout<<"Saved "<<cnt<<" traps to "<<fn
             <<" ("<<humanBytes(f.tellp())<<")\n";
}
bool loadDPBinary(const std::string& fn){
    std::ifstream f(fn,std::ios::binary|std::ios::ate);
    if(!f){ std::cerr<<"[ERR] open "<<fn<<"\n"; return false; }
    if(f.tellg()%sizeof(DpItem)){ std::cerr<<"[ERR] bad size\n"; return false; }
    f.seekg(0);
    DpItem it; uint64_t n = 0;
    while(f.read(reinterpret_cast<char*>(&it),sizeof(it))){
        Int p; p.Set32Bytes(it.priv);
        if(dp_insert_unique(it.fp,p)) bloom->Add(uint32_t(it.fp));
        if((++n & 0xFFFFF) == 0) std::cout<<"\rLoaded "<<n<<std::flush;
    }
    std::cout<<"\rLoaded "<<n<<" traps (done)\n";
    return true;
}

// ─── Table cache (DP_table + st_used + bloom) ─────────────────────────────────
#pragma pack(push,1)
struct TableMetaHeader{
    char     magic[8];      // "MK1TAB1"
    uint32_t version;       // 1
    uint32_t dp_bits;
    uint32_t k;
    uint32_t reserved;
    uint64_t traps;
    uint64_t cap;
    uint64_t usedCount;
    uint64_t bloomBytes;
    uint64_t bloomSeed;
    uint8_t  A[32];
    uint8_t  B[32];
};
#pragma pack(pop)

static inline void IntTo32Bytes(const Int& v,uint8_t out[32]){
    Int tmp; intCopy(tmp,v); tmp.Get32Bytes(out);
}
static inline void BytesToInt32(const uint8_t in[32],Int& out){
    out.Set32Bytes(const_cast<unsigned char*>(in)); // Int API не const-friendly
}
static inline uint64_t computeBloomSeed(const Int& A,const Int& B,
                                        unsigned dp_bits,unsigned k,
                                        uint64_t traps,size_t cap){
    uint64_t x = 0xB10FB10FB10FB10FULL;
    x ^= splitmix64(IntLow64(A));
    x ^= splitmix64(IntLow64(B));
    x ^= splitmix64((uint64_t(dp_bits) << 32) ^ uint64_t(k));
    x ^= splitmix64(traps);
    x ^= splitmix64(uint64_t(cap));
    return splitmix64(x);
}

static bool saveTableCache(const std::string& tablePath,
                           const Int& A,const Int& B,
                           unsigned dp_bits,unsigned k,
                           uint64_t traps,uint64_t bloomSeed){
    const std::string metaPath = tablePath + ".meta";
    TableMetaHeader h{};
    std::memcpy(h.magic,"MK1TAB1",7);
    h.magic[7]=0;
    h.version   = 1;
    h.dp_bits   = dp_bits;
    h.k         = k;
    h.traps     = traps;
    h.cap       = dp.cap;
    h.usedCount = dp.size.load(std::memory_order_relaxed);
    h.bloomBytes= bloom->SizeInBytes();
    h.bloomSeed = bloomSeed;
    IntTo32Bytes(A,h.A);
    IntTo32Bytes(B,h.B);

    std::ofstream f(metaPath,std::ios::binary|std::ios::trunc);
    if(!f){ std::cerr<<"[ERR] open "<<metaPath<<"\n"; return false; }
    f.write(reinterpret_cast<char*>(&h),sizeof(h));

    const size_t bitBytes = (dp.cap + 7) / 8;
    std::vector<uint8_t> used(bitBytes);

#pragma omp parallel for schedule(static)
    for(size_t b=0;b<bitBytes;++b){
        uint8_t v=0;
        const size_t base=b*8;
        for(int j=0;j<8;++j){
            const size_t idx=base + size_t(j);
            if(idx < dp.cap &&
               dp.st_used[idx].load(std::memory_order_relaxed))
                v |= uint8_t(1u<<j);
        }
        used[b]=v;
    }

    f.write(reinterpret_cast<const char*>(used.data()),used.size());
    f.write(reinterpret_cast<const char*>(bloom->data()),h.bloomBytes);

    if(!f){ std::cerr<<"[ERR] write "<<metaPath<<"\n"; return false; }
    std::cout<<"Saved table cache: "<<metaPath
             <<" | used="<<h.usedCount
             <<" | bloom="<<humanBytes(h.bloomBytes)<<"\n";
    return true;
}

static bool loadTableCache(const std::string& tablePath,
                           const Int& A,const Int& B,
                           unsigned dp_bits,unsigned k,
                           uint64_t traps,uint64_t bloomSeed){
    const std::string metaPath = tablePath + ".meta";
    std::ifstream f(metaPath,std::ios::binary);
    if(!f){ std::cerr<<"[ERR] missing "<<metaPath<<"\n"; return false; }

    TableMetaHeader h{};
    f.read(reinterpret_cast<char*>(&h),sizeof(h));
    if(!f){ std::cerr<<"[ERR] read header "<<metaPath<<"\n"; return false; }
    if(std::memcmp(h.magic,"MK1TAB1",7)!=0 || h.version!=1){
        std::cerr<<"[ERR] bad meta magic/version\n"; return false;
    }
    if(h.cap != dp.cap){
        std::cerr<<"[ERR] meta cap mismatch (meta "<<h.cap<<" vs need "<<dp.cap<<")\n"; return false;
    }
    if(h.dp_bits != dp_bits || h.k != k){
        std::cerr<<"[ERR] meta dp_bits/k mismatch\n"; return false;
    }
    if(h.traps != traps){
        std::cerr<<"[ERR] meta traps mismatch\n"; return false;
    }
    if(h.bloomBytes != bloom->SizeInBytes()){
        std::cerr<<"[ERR] meta bloomBytes mismatch\n"; return false;
    }
    if(h.bloomSeed != bloomSeed){
        std::cerr<<"[ERR] meta bloomSeed mismatch\n"; return false;
    }
    Int A0,B0; BytesToInt32(h.A,A0); BytesToInt32(h.B,B0);
    if(!A0.IsEqual(&const_cast<Int&>(A)) || !B0.IsEqual(&const_cast<Int&>(B))){
        std::cerr<<"[ERR] meta range mismatch\n"; return false;
    }

    const size_t bitBytes = (dp.cap + 7) / 8;
    std::vector<uint8_t> used(bitBytes);
    f.read(reinterpret_cast<char*>(used.data()),used.size());
    if(!f){ std::cerr<<"[ERR] read st_used "<<metaPath<<"\n"; return false; }

#pragma omp parallel for schedule(static)
    for(size_t i=0;i<dp.cap;++i){
        const uint8_t byte = used[i>>3];
        const uint8_t v = (byte >> (i & 7)) & 1;
        dp.st_used[i].store(v,std::memory_order_relaxed);
        dp.st_lock[i].store(0,std::memory_order_relaxed);
    }

    f.read(reinterpret_cast<char*>(bloom->data()),h.bloomBytes);
    if(!f){ std::cerr<<"[ERR] read bloom "<<metaPath<<"\n"; return false; }

    dp.size.store(h.usedCount,std::memory_order_relaxed);
    dpDone.store(std::min<uint64_t>(traps,h.usedCount),std::memory_order_relaxed);

    std::cout<<"Loaded table cache: "<<metaPath
             <<" | used="<<h.usedCount
             <<" | bloom="<<humanBytes(h.bloomBytes)<<"\n";
    return true;
}



static bool readTableMetaHeader(const std::string& tablePath,TableMetaHeader& h){
    const std::string metaPath = tablePath + ".meta";
    std::ifstream f(metaPath,std::ios::binary);
    if(!f) return false;
    f.read(reinterpret_cast<char*>(&h),sizeof(h));
    return f && std::memcmp(h.magic,"MK1TAB1",7)==0 && h.version==1;
}
static bool resumeMetaMatches(const TableMetaHeader& h,
                              const Int& A,const Int& B,
                              unsigned bits,unsigned k){
    if(h.dp_bits != bits || h.k != k) return false;
    Int A0,B0; BytesToInt32(h.A,A0); BytesToInt32(h.B,B0);
    return A0.IsEqual(&const_cast<Int&>(A)) &&
           B0.IsEqual(&const_cast<Int&>(B));
}
static inline bool scalarIsZero(const Scalar256& s){
    return (s.w[0] | s.w[1] | s.w[2] | s.w[3]) == 0;
}
static inline uint64_t dpMaskForBits(unsigned bits){
    return bits >= 64 ? ~0ULL : ((1ULL<<bits)-1);
}
static bool validateTrapSlot(const DPSlot& slot,unsigned bits,Int& outKey){
    if(slot.fp == 0 && scalarIsZero(slot.key)) return false;

    scalarToInt(slot.key,outKey);
    if(outKey.IsZero()) return false;
    if(intGE(outKey,ORDER_N)) return false;

    Point p = mulP(outKey);
    if((IntLow64(p.x) & dpMaskForBits(bits)) != 0) return false;

    fp_t expect = splitmix64(IntLow64(p.x) ^ uint64_t(!p.y.IsEven()));
    return expect == slot.fp;
}
static size_t roundUpPow2(size_t v){
    size_t p = 1;
    while(p < v) p <<= 1;
    return p;
}
static std::string makeUniquePath(const std::string& base){
    if(::access(base.c_str(),F_OK) != 0) return base;
    for(unsigned i=1;i<10000;++i){
        std::string p = base + "." + std::to_string(i);
        if(::access(p.c_str(),F_OK) != 0) return p;
    }
    return base + ".fallback";
}
static bool promoteResumedTable(const std::string& workPath,
                                const std::string& finalPath){
    std::string backupPath = makeUniquePath(finalPath + ".resume.bak");

    bool hadFinal = (::access(finalPath.c_str(),F_OK) == 0);
    bool hadMeta  = (::access((finalPath + ".meta").c_str(),F_OK) == 0);

    if(hadFinal){
        if(::rename(finalPath.c_str(),backupPath.c_str()) != 0){
            perror("rename(old_table)");
            return false;
        }
        if(hadMeta){
            (void)::rename((finalPath + ".meta").c_str(),
                           (backupPath + ".meta").c_str());
        }
        std::cout<<"Resume: original table moved to "<<backupPath<<"\n";
    }

    if(::rename(workPath.c_str(),finalPath.c_str()) != 0){
        perror("rename(new_table)");
        if(hadFinal) (void)::rename(backupPath.c_str(),finalPath.c_str());
        if(hadMeta)  (void)::rename((backupPath + ".meta").c_str(),
                                    (finalPath + ".meta").c_str());
        return false;
    }

    std::cout<<"Resume: repaired table promoted to "<<finalPath<<"\n";
    return true;
}
static bool resumeScanTable(const std::string& srcPath,unsigned bits){
    std::ifstream f(srcPath,std::ios::binary|std::ios::ate);
    if(!f){ std::cerr<<"[ERR] open "<<srcPath<<"\n"; return false; }

    std::streamoff totalBytesOff = f.tellg();
    if(totalBytesOff < 0){
        std::cerr<<"[ERR] tellg "<<srcPath<<"\n";
        return false;
    }
    const uint64_t totalBytes = uint64_t(totalBytesOff);
    const uint64_t totalSlots = totalBytes / sizeof(DPSlot);
    const uint64_t tailBytes  = totalBytes % sizeof(DPSlot);

    f.seekg(0);

    uint64_t scanned=0, empty=0, bad=0, dup=0;
    auto lastStat = std::chrono::steady_clock::now();

    while(scanned < totalSlots){
        DPSlot slot{};
        f.read(reinterpret_cast<char*>(&slot),sizeof(slot));
        if(!f){
            std::cerr<<"[ERR] read "<<srcPath<<" at slot "<<scanned<<"\n";
            return false;
        }
        ++scanned;

        if(slot.fp == 0 && scalarIsZero(slot.key)){
            ++empty;
        }else{
            Int key;
            if(!validateTrapSlot(slot,bits,key)){
                ++bad;
            }else{
                if(dp_insert_unique(slot.fp,key)){
                    bloom->Add(uint32_t(slot.fp));
                }else{
                    ++dup;
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        if(now - lastStat >= std::chrono::seconds(5)){
            uint64_t good = dpDone.load(std::memory_order_relaxed);
            double pct = totalSlots ? (100.0 * double(scanned) / double(totalSlots)) : 100.0;
            std::cout<<"\rResume scan: "<<scanned<<"/"<<totalSlots
                     <<" ("<<std::fixed<<std::setprecision(2)<<pct<<"%)"
                     <<" | valid "<<good
                     <<" | dup "<<dup
                     <<" | bad "<<bad
                     <<" | empty "<<empty
                     <<std::flush;
            lastStat = now;
        }
    }

    if(tailBytes){
        std::cout<<"\nResume: ignored trailing partial tail of "<<tailBytes<<" bytes\n";
    }

    std::cout<<"\rResume scan: "<<scanned<<"/"<<totalSlots
             <<" (100.00%)"
             <<" | valid "<<dpDone.load(std::memory_order_relaxed)
             <<" | dup "<<dup
             <<" | bad "<<bad
             <<" | empty "<<empty
             <<" (done)\n";
    return true;
}
// ─── range split ──────────────────────────────────────────────────────────────
struct RangeSeg{ Int start,length; };
static std::vector<RangeSeg>
splitRange(const Int& A,const Int& total,unsigned parts){
    std::vector<RangeSeg> seg(parts);
    Int chunk(total); Int div((uint64_t)parts); chunk.Div(&div,nullptr);
    Int lenLast(total);
    if(parts>1){ Int t(chunk); Int m((uint64_t)(parts-1)); t.Mult(&m); lenLast.Sub(&t); }
    for(unsigned i=0;i<parts;++i){
        seg[i].start = A;
        if(i){
            Int off(chunk); Int k((uint64_t)i); off.Mult(&k); seg[i].start.Add(&off);
        }
        seg[i].length = (i==parts-1)?lenLast:chunk;
    }
    return seg;
}

// ─── xoshiro rng ──────────────────────────────────────────────────────────────
struct xoshiro256ss {
    uint64_t s[4];

    static inline uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    explicit xoshiro256ss(uint64_t seed = 1) {
        for (int i = 0; i < 4; ++i) {
            seed = splitmix64(seed);
            s[i] = seed;
        }
    }

    inline uint64_t operator()() noexcept { return next(); }

    inline uint64_t next() noexcept {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];  s[3] ^= s[1];
        s[1] ^= s[2];  s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    using result_type = uint64_t;
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return ~0ULL; }
};

// ─── batch-EC-add ─────────────────────────────────────────────────────────────
template<unsigned N>
static inline void batchAdd(Point* base,Point* plus){
    std::array<Int,N> dX;
    for(unsigned i=0;i<N;++i) dX[i].ModSub(&plus[i].x,&base[i].x);
    static thread_local IntGroup grp(N); grp.Set(dX.data()); grp.ModInv();

    for(unsigned i=0;i<N;++i){
        Int dY; dY.ModSub(&plus[i].y,&base[i].y);
        Int k ; k .ModMulK1(&dY,&dX[i]);
        Int k2; k2.ModSquareK1(&k);
        Int xn(base[i].x); xn.ModNeg(); xn.ModAdd(&k2); xn.ModSub(&plus[i].x);
        Int dx(base[i].x); dx.ModSub(&xn); dx.ModMulK1(&k);
        base[i].x = xn;
        base[i].y.ModNeg();
        base[i].y.ModAdd(&dx);
    }
}

// ─── jump-table ───────────────────────────────────────────────────────────────
static std::vector<Point> jumps;
static void buildJumpTable(unsigned k){
    jumps.resize(k);
#pragma omp parallel for schedule(static)
    for(unsigned i=0;i<k;++i){
        Int e((uint64_t)1); e.ShiftL(int(i+1));
        jumps[i] = mulP(e);
    }
}

// ─── globals ──────────────────────────────────────────────────────────────────
static std::atomic<uint64_t> hops{0}, restarts{0};
static std::atomic<bool>     solved{false};
static Int  privFound;
static std::atomic<unsigned> found_tid{0};
static std::atomic<uint64_t> winner_wraps{0};
static std::once_flag        record_flag;

// ─── wrap helper ──────────────────────────────────────────────────────────────
static inline void addWrapCnt(Int& v,const Int& d,const Int& len,uint64_t& wraps){
    v.Add(&const_cast<Int&>(d));
    if(intGE(v,len)){ v.Sub(&const_cast<Int&>(len)); ++wraps; }
}

// ─── Traps (Phase-1) ─────────────────────────────────────────────────────────
static constexpr unsigned K_DP = 512;
static void buildDP_segment(const RangeSeg& seg,uint64_t target,
                            unsigned k,unsigned bits,uint64_t seed){
    const uint64_t mask = (1ULL<<bits)-1;
    xoshiro256ss rng(seed);
    std::uniform_int_distribution<uint64_t> rd;

    std::array<Int,   K_DP> dist;
    std::array<uint64_t,K_DP> wraps{};
    std::array<Point, K_DP> cur, stepPts;

    const size_t BATCH_SIZE = 256;
    std::vector<std::pair<fp_t,Int>> batch;
    batch.reserve(BATCH_SIZE);

    auto rndMod=[&](Int& o){
        o.SetInt32(0); int parts=(bitlen(seg.length)+63)/64;
        for(int p=0;p<parts;++p){
            Int t((uint64_t)rd(rng)); t.ShiftL(p*64); o.Add(&t);
        }
        o.Mod(&const_cast<Int&>(seg.length));
    };
    for(unsigned i=0;i<K_DP;++i){
        rndMod(dist[i]); Int a(seg.start); a.Add(&dist[i]); cur[i] = mulP(a);
    }

    uint64_t made = 0;
    while(made < target){
        for(unsigned i=0;i<K_DP;++i){
            uint64_t h = splitmix64(IntLow64(cur[i].x)) % k;
            Int step((uint64_t)1); step.ShiftL(int(h+1));

            if((IntLow64(cur[i].x) & mask) == 0){
                fp_t fp = splitmix64(IntLow64(cur[i].x) ^
                                     uint64_t(!cur[i].y.IsEven()));

                Int scalar(seg.length);
                Int w((uint64_t)wraps[i]); scalar.Mult(&w);
                scalar.Add(&const_cast<Int&>(dist[i]));
                scalar.Add(&const_cast<Int&>(seg.start));
                scalar.Mod(&ORDER_N);

                batch.emplace_back(fp,scalar);
                if(batch.size() >= BATCH_SIZE || made + batch.size() >= target){
#pragma omp critical(dp_insert)
                    {
                        for(auto& it: batch){
                            if(dp_insert_unique(it.first,it.second)){
                                bloom->Add(uint32_t(it.first));
                                ++made;
                                if(made==target) break;
                            }
                        }
                        batch.clear();
                    }
                }
            }

            stepPts[i] = jumps[h];
            addWrapCnt(dist[i],step,seg.length,wraps[i]);
        }
        batchAdd<K_DP>(cur.data(),stepPts.data());
    }
    if(!batch.empty()){
#pragma omp critical(dp_insert)
        {
            for(auto& it: batch){
                if(dp_insert_unique(it.first,it.second)){
                    bloom->Add(uint32_t(it.first));
                    ++made;
                }
            }
            batch.clear();
        }
    }
}

// ─── Phase-2: wild kangaroos ─────────────────────────────────────────────────
static constexpr unsigned K   = 512;
static constexpr unsigned CACHE_LIMIT = 1024;

struct PendingCheck{ fp_t fp; unsigned idx; };

static void worker(uint32_t tid,const RangeSeg& seg,const Point& pub,
                   unsigned k,unsigned bits){
    struct LoopDet{
        uint64_t next,cnt,sig;
        void reset(uint64_t s){ next=1024; cnt=0; sig=s; }
    };

    const uint64_t mask = (1ULL<<bits)-1;
    std::mt19937_64 rng(splitmix64(0xDEADBEEF*tid));
    std::uniform_int_distribution<uint64_t> rd;

    std::array<Int,   K> dist;
    std::array<uint64_t,K> wraps{};
    std::array<Point, K> cur, stepPts;
    std::array<LoopDet,K> loop;

    auto rndMod=[&](Int& o){
        o.SetInt32(0); int parts=(bitlen(seg.length)+63)/64;
        for(int p=0;p<parts;++p){
            Int t((uint64_t)rd(rng)); t.ShiftL(p*64); o.Add(&t);
        }
        o.Mod(&const_cast<Int&>(seg.length));
    };
    for(unsigned i=0;i<K;++i){
        rndMod(dist[i]);
        cur[i]  = addP(pub,mulP(dist[i]));
        loop[i].reset(splitmix64(IntLow64(cur[i].x)^uint64_t(!cur[i].y.IsEven())));
    }

    madvise(dp.slots,dp.mapBytes,MADV_SEQUENTIAL);

    uint64_t local=0; const uint64_t FLUSH = 1ULL<<18; 
    std::vector<PendingCheck> cache; cache.reserve(CACHE_LIMIT);

    while(!solved.load()){
        for(unsigned i=0;i<K;++i){
            if(solved.load()) return;

            uint64_t x64 = IntLow64(cur[i].x);
            uint64_t h   = splitmix64(x64)%k;

            auto& ld = loop[i];
            if(++ld.cnt == ld.next){
                uint64_t sig = splitmix64(x64^uint64_t(!cur[i].y.IsEven()));
                if(sig == ld.sig){
                    rndMod(dist[i]);
                    cur[i]  = addP(pub,mulP(dist[i]));
                    wraps[i]=0; ld.reset(sig);
                    restarts.fetch_add(1,std::memory_order_relaxed);
                    continue;
                }
                ld.sig = sig; ld.next <<= 1;
            }

            stepPts[i] = jumps[h];
            Int step((uint64_t)1); step.ShiftL(int(h+1));
            addWrapCnt(dist[i],step,seg.length,wraps[i]);
            ++local;
        }
        batchAdd<K>(cur.data(),stepPts.data());
        if(local >= FLUSH){ hops.fetch_add(local); local = 0; }

        for(unsigned i=0;i<K;++i){
            if(solved.load()) return;
            if((IntLow64(cur[i].x)&mask)!=0) continue;

            fp_t fp = splitmix64(IntLow64(cur[i].x)^uint64_t(!cur[i].y.IsEven()));
            cache.push_back({fp,i});

            if(cache.size() >= CACHE_LIMIT){
#pragma omp critical(dp_query)
                {
                    for(auto& item: cache){
                        if(!bloom->Find(uint32_t(item.fp))) continue;
                        Int trap;
                        if(!dp_find(item.fp,trap)) continue;

                        Int dw(seg.length);
                        Int w((uint64_t)wraps[item.idx]); dw.Mult(&w);
                        dw.Add(&const_cast<Int&>(dist[item.idx]));
                        dw.Mod(&ORDER_N);

                        Int priv; intCopy(priv,trap); priv.Sub(&dw); priv.Mod(&ORDER_N);
                        Point tst = mulP(priv);
                        if(tst.x.IsEqual(&const_cast<Int&>(pub.x)) &&
                           tst.y.IsEqual(&const_cast<Int&>(pub.y)))
                        {
                            std::call_once(record_flag,[]{});
                            intCopy(privFound,priv);
                            found_tid.store(tid);
                            winner_wraps.store(wraps[item.idx]);
                            solved.store(true);
                        }
                    }
                }
                cache.clear();
                if(solved.load()) return;
            }
        }
    }
    if(local) hops.fetch_add(local);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc,char** argv)
{
    /* init curve */
    P_PRIME.SetBase16((char*)P_HEX); ORDER_N.SetBase16((char*)N_HEX); secp.Init();

    /* ── CLI ── */
    Int A,B; uint64_t traps=0; unsigned bits=12; size_t ramGB=8;
    Point pub; unsigned k_user=0; bool saveDP=false, loadDP=false;
    bool useTable=false, saveTable=false, resumeTable=false;
    std::string tablePath="dp_table.bin", tableWorkPath;
    std::string dpFile;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--range"){ std::string s=argv[++i]; auto p=s.find(':');
            A.SetBase10((char*)s.substr(0,p).c_str());
            B.SetBase10((char*)s.substr(p+1).c_str());
        }else if(a=="--dp_point") traps=std::strtoull(argv[++i],nullptr,10);
        else if(a=="--dp_bits")  bits=std::stoul(argv[++i]);
        else if(a=="--ram"||a=="--ram-limit") ramGB=std::stoull(argv[++i]);
        else if(a=="--k") k_user=std::stoul(argv[++i]);
        else if(a=="--pubkey"){
            std::string h=argv[++i]; if(h.rfind("0x",0)==0) h.erase(0,2);
            char pc=h[1]; Int x; x.SetBase16((char*)h.substr(2).c_str());
            pub.x=x; pub.y=secp.GetY(x,pc=='2');
        }else if(a=="--save-dp"||a=="-s") saveDP=true;
        else if(a=="--use-table"){ useTable=true; tablePath=argv[++i]; }
        else if(a=="--resume-table"){ resumeTable=true; tablePath=argv[++i]; }
        else if(a=="--save-table"){ saveTable=true; tablePath=argv[++i]; }
        else if(a=="--table"){ tablePath=argv[++i]; }
        else if(a=="--load-dp"){ loadDP=true; dpFile=argv[++i]; }
        else{ std::cerr<<"Unknown "<<a<<'\n'; return 1; }
    }
    if(A.IsZero()||B.IsZero()){ std::cerr<<"--range missing\n"; return 1; }

    /* ── params ── */
    Int range(B); range.Sub(&A);
    unsigned Lbits=range.GetBitLength();
    if(!traps){
        traps=(Lbits>=52)?(1ULL<<(Lbits/2))
             : uint64_t(std::ceil(range.ToDouble()/std::sqrt(range.ToDouble())));
    }
    unsigned k = k_user? k_user : std::max(1u,Lbits/2);

    /* новые параметры */
    constexpr double MAX_LOAD     = 0.50; 
    constexpr double bloomFactor  = 10.0;  

    size_t cap0 = size_t(std::ceil(double(traps) / MAX_LOAD));
    size_t cap  = 1;
    while(cap < cap0) cap <<= 1;

    TableMetaHeader resumeMeta{};
    bool resumeMetaFound = false, resumeMetaOk = false;
    if(resumeTable){
        struct stat st{};
        if(::stat(tablePath.c_str(),&st) != 0){
            perror("stat(resume-table)");
            return 1;
        }

        uint64_t srcBytes = (st.st_size > 0) ? uint64_t(st.st_size) : 0;
        size_t srcSlots = size_t(srcBytes / sizeof(DPSlot));
        if(srcSlots) cap = std::max(cap, roundUpPow2(srcSlots));

        resumeMetaFound = readTableMetaHeader(tablePath,resumeMeta);
        if(resumeMetaFound){
            resumeMetaOk = resumeMetaMatches(resumeMeta,A,B,bits,k);
            if(!resumeMetaOk){
                std::cerr<<"[ERR] existing .meta does not match current --range/--dp_bits/--k\n";
                return 1;
            }
            if(resumeMeta.cap) cap = std::max(cap, size_t(resumeMeta.cap));
        }

        tableWorkPath = makeUniquePath(tablePath + ".resume.new");
    }else{
        tableWorkPath = tablePath;
    }

    dp.init(tableWorkPath,cap);

    size_t bloomBytes=size_t(traps*bloomFactor);
    uint64_t bloomSeed = computeBloomSeed(A,B,bits,k,traps,cap);
    std::cout<<"\n=========== Phase-0: Data summary ==========\n";
    std::cout<<"DP table (SSD): "<<humanBytes(cap*sizeof(DPSlot))
             <<"  ( "<<traps<<" / "<<cap<<" slots, load "
             <<std::fixed<<std::setprecision(2)
             <<double(traps)/cap*100<<"% )\n";
    std::cout<<"Bloom    (RAM): "<<humanBytes(bloomBytes)<<'\n';

    if(resumeTable){
        std::cout<<"Resume source: "<<tablePath<<"\n";
        if(resumeMetaFound){
            std::cout<<"Resume meta  : found"
                     <<(resumeMetaOk ? " [compatible]\n" : " [mismatch]\n");
        }else{
            std::cout<<"Resume meta  : not found\n";
        }
    }

    bloom=new simd_bloom::SimdBlockFilterFixed<>(bloomBytes, bloomSeed);

    unsigned th=std::max(1u,std::thread::hardware_concurrency());
    auto segs=splitRange(A,range,th);
    buildJumpTable(k);

    auto runPhase1Build = [&](uint64_t need){
        if(!need) return;

        uint64_t per = (need + th - 1) / th;
        std::thread progress([&]{
            using clock = std::chrono::steady_clock;
            auto t0      = clock::now();
            auto lastT   = t0;
            uint64_t lastCnt  = dpDone.load(std::memory_order_relaxed);
            uint64_t startCnt = lastCnt;

            while(true){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));

                uint64_t cur = dpDone.load(std::memory_order_relaxed);
                if(cur >= traps) break;

                auto nowT = clock::now();
                if(nowT - lastT < std::chrono::seconds(5))
                    continue;

                double dt  = std::chrono::duration<double>(nowT - lastT).count();
                double tot = std::chrono::duration<double>(nowT - t0).count();
                uint64_t d = cur - lastCnt;
                uint64_t built = cur - startCnt;

                double instRate = (dt  > 0.0) ? (double)d    / dt  : 0.0;
                double avgRate  = (tot > 0.0) ? (double)built / tot : 0.0;
                double pct      = (traps > 0) ? (double)cur * 100.0 / (double)traps : 0.0;

                uint64_t etaSec = 0;
                if(avgRate > 0.0 && cur < traps){
                    etaSec = (uint64_t)((double)(traps - cur) / avgRate);
                }
                uint64_t hh = etaSec / 3600;
                uint64_t mm = (etaSec / 60) % 60;
                uint64_t ss = etaSec % 60;

                std::ostringstream line;
                line << "\rUnique traps: " << cur << '/' << traps
                     << " (" << std::fixed << std::setprecision(2) << pct << "%)"
                     << " | " << std::fixed << std::setprecision(0) << instRate << " traps/s"
                     << " | ETA: " << hh << ':' << std::setw(2) << std::setfill('0') << mm
                     << ':' << std::setw(2) << ss << std::setfill(' ')
                     << "        ";

                std::cout << line.str() << std::flush;

                lastT = nowT;
                lastCnt = cur;
            }

            std::cout << "\rUnique traps: " << traps << "/" << traps << " (done)\n";
        });

#pragma omp parallel for schedule(static)
        for(unsigned t=0;t<th;++t)
            buildDP_segment(segs[t],per,k,bits,
                            splitmix64(0xABCDEF12345678ULL^t));

        progress.join();
    };

    // ─── Phase-1 ────────────────────────────────────────────────────────────
    dp.enable_flush.store(false);
    if(resumeTable){
        std::cout<<"\n========== Phase-1: Resuming table =========\n";

        if(!resumeScanTable(tablePath,bits)) return 1;

        uint64_t recovered = dpDone.load(std::memory_order_relaxed);
        std::cout<<"Resume: recovered "<<recovered<<" unique traps from table\n";

        if(recovered < traps){
            std::cout<<"Resume: continuing Phase-1 for remaining "
                     <<(traps - recovered)<<" traps\n";
            runPhase1Build(traps - recovered);
        }else{
            std::cout<<"Resume: target already satisfied, skipping trap build\n";
        }

        dp.fullSync();
        if(!promoteResumedTable(tableWorkPath,tablePath)) return 1;
        if(!saveTableCache(tablePath,A,B,bits,k,traps,bloomSeed)) return 1;
        if(saveDP) saveDPBinary("DP.bin");

    }else if(useTable){
        std::cout<<"\n========== Phase-1: Reusing DP table =======\n";
        if(!loadTableCache(tablePath,A,B,bits,k,traps,bloomSeed)) return 1;

    }else{
        std::cout<<"\n========== Phase-1: Building traps =========\n";
        if(loadDP){
            if(!loadDPBinary(dpFile)) return 1;
        }else{
            runPhase1Build(traps);
            if(saveDP) saveDPBinary("DP.bin");
        }

        dp.fullSync();
        if(saveTable){
            if(!saveTableCache(tablePath,A,B,bits,k,traps,bloomSeed)) return 1;
        }
    }
    dp.enable_flush.store(true);

    // ─── Phase-2 ────────────────────────────────────────────────────────────
    std::cout<<"\n=========== Phase-2: Kangaroos =============\n";
    auto t0=std::chrono::steady_clock::now();
    std::thread pool([&]{
#pragma omp parallel for num_threads(th) schedule(static)
        for(unsigned id=0;id<th;++id) worker(id,segs[id],pub,k,bits);
    });

    uint64_t lastHops=0;
    auto lastStat=t0;
    while(true){
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if(solved.load()) break;
        auto nowTick=std::chrono::steady_clock::now();
        if(nowTick-lastStat<std::chrono::seconds(5)) continue;

        double dt=std::chrono::duration<double>(nowTick-lastStat).count();
        lastStat=nowTick;
        uint64_t now=hops.load(); uint64_t delta=now-lastHops; lastHops=now;
        double sp=delta/dt; const char* u=" H/s";
        if(sp>1e6){ sp/=1e6; u=" MH/s";}
        else if(sp>1e3){ sp/=1e3; u=" kH/s";}
        uint64_t sec=std::chrono::duration_cast<std::chrono::seconds>(nowTick-t0).count();

        std::cout<<"\rSpeed: "<<std::fixed<<std::setprecision(2)<<sp<<u
                 <<" | Hops: "<<now
                 <<" | Restart wild: "<<restarts.load()
                 <<" | Time: "<<sec/3600<<':'<<std::setw(2)<<std::setfill('0')
                 <<(sec/60)%60<<':'<<std::setw(2)<<sec%60<<std::flush;
    }
    pool.join();

    // ─── Phase-3 ────────────────────────────────────────────────────────────
    auto msTot=std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now()-t0).count();
    uint64_t h=msTot/3'600'000,m=(msTot/60'000)%60,s=(msTot/1'000)%60,ms=msTot%1'000;

    std::cout<<"\n\n============= Phase-3: Result ==============\n";
    std::cout<<"Private key : 0x"<<std::setw(64)<<std::setfill('0')
             <<privFound.GetBase16()<<"\n";
    std::cout<<"Found by thr: "<<found_tid.load()<<"\n";
    std::cout<<"Wild wraps  : "<<winner_wraps.load()
             <<(winner_wraps.load()?"  [wrapped]\n":"  [no wrap]\n");
    std::cout<<"Wild restart: "<<restarts.load()<<"\n";
    std::cout<<"Total time  : "<<std::setw(2)<<h<<':'<<std::setw(2)<<m<<':'
             <<std::setw(2)<<s<<'.'<<std::setw(3)<<ms<<"\n";

    { std::ofstream("FOUND.txt")<<"0x"<<std::setw(64)<<std::setfill('0')
                                <<privFound.GetBase16()<<"\n"; }
    std::cout<<"Private key : saved to FOUND.txt\n";

    delete bloom; dp.close();
    return 0;
}


