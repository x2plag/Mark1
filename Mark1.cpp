/***************************************************************************************************
 * Pollard–Kangaroo  (wrap-aware, user-configurable k, live counter, loop detector, restart counter)
 * Coded by DooKoo2
 * Load/Save DP tech by NoMachine
 *
 *  g++ Mark1.cpp Int.cpp SECP256K1.cpp Point.cpp Random.cpp IntMod.cpp IntGroup.cpp Timer.cpp -O3 -march=native -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-
 *    interposition -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -fipa-modref -flto -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -std=c++17 -fopenmp -pthread -o Mark1
 *
 ***************************************************************************************************/
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
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
#include <omp.h>

// --------------------------- project libs ------------------------------------
#include "Int.h"
#include "Point.h"
#include "SECP256K1.h"
#include "IntGroup.h"
#include "simd_block_bloom.h"

// ─── Scalar256 ────────────────────────────────────────────────────────────────
struct Scalar256 { uint64_t w[4]; };

static inline void intToScalar(const Int &src, Scalar256 &dst){
    dst.w[0]=src.bits64[0]; dst.w[1]=src.bits64[1];
    dst.w[2]=src.bits64[2]; dst.w[3]=src.bits64[3];
}
static inline void scalarToInt(const Scalar256 &s, Int &dst){
    dst.SetInt32(0);
    for(int i=3;i>=0;--i){
        dst.ShiftL(64);
        Int part(s.w[i]); dst.Add(&part);
    }
}

// ─── Int helpers ──────────────────────────────────────────────────────────────
static inline void intCopy(Int &d,const Int &s){ d.Set(const_cast<Int*>(&s)); }
static inline bool intGE(const Int &a,const Int &b){
    return const_cast<Int&>(a).IsGreaterOrEqual(const_cast<Int*>(&b));
}
static inline uint64_t IntLow64(const Int &n){ return n.bits64[0]; }
static inline int      bitlen (const Int &v){ return const_cast<Int&>(v).GetBitLength(); }

// ─── misc ─────────────────────────────────────────────────────────────────────
static inline uint64_t splitmix64(uint64_t x){
    x+=0x9E3779B97F4A7C15ULL;
    x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
    x=(x^(x>>27))*0x94D049BB133111EBULL;
    return x^(x>>31);
}
static Int hexToInt(const std::string& h){ Int x; x.SetBase16((char*)h.c_str()); return x; }
static Int decToInt(const std::string& d){ Int x; x.SetBase10((char*)d.c_str()); return x; }
static std::string intHex(const Int &v,bool pad=false){
    Int t; intCopy(t,v); std::string s=t.GetBase16();
    if(pad && s.size()<64) s.insert(0,64-s.size(),'0');
    return s;
}
static std::string humanBytes(size_t bytes){
    constexpr const char* unit[]{"B","Kb","Mb","Gb","Tb"};
    double v=double(bytes); int u=0; while(v>=1024.0&&u<4){v/=1024.0;++u;}
    std::ostringstream o;
    if(v<10)       o<<std::fixed<<std::setprecision(2);
    else if(v<100) o<<std::fixed<<std::setprecision(1);
    else           o<<std::fixed<<std::setprecision(0);
    o<<v<<unit[u]; return o.str();
}

// ─── curve ────────────────────────────────────────────────────────────────────
static const char *P_HEX="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
static const char *N_HEX="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
static Int P_PRIME, ORDER_N; static Secp256K1 secp;

static inline Point addP(const Point &a,const Point &b){
    if(const_cast<Int&>(a.x).IsZero()&&const_cast<Int&>(a.y).IsZero()) return b;
    if(const_cast<Int&>(b.x).IsZero()&&const_cast<Int&>(b.y).IsZero()) return a;
    return secp.AddDirect(const_cast<Point&>(a),const_cast<Point&>(b));
}
static inline Point mulP(const Int &k){ Int t; intCopy(t,k); return secp.ComputePublicKey(&t); }

// ─── range split ──────────────────────────────────────────────────────────────
struct RangeSeg{ Int start,length; };

static std::vector<RangeSeg> splitRange(const Int &A,const Int &total,unsigned parts){
    std::vector<RangeSeg> seg(parts);
    Int chunk(total); Int div((uint64_t)parts); chunk.Div(&div,nullptr);
    Int lenLast(total);
    if(parts>1){ Int t(chunk); Int m((uint64_t)(parts-1)); t.Mult(&m); lenLast.Sub(&t); }
    for(unsigned i=0;i<parts;++i){
        seg[i].start=A;
        if(i){ Int off(chunk); Int k((uint64_t)i); off.Mult(&k); seg[i].start.Add(&off); }
        seg[i].length=(i==parts-1)?lenLast:chunk;
    }
    return seg;
}

// ─── wrap helper ──────────────────────────────────────────────────────────────
static inline void addWrapCnt(Int &v,const Int &d,const Int &len,uint64_t &wraps){
    v.Add(const_cast<Int*>(&d));
    if(intGE(v,len)){ v.Sub(const_cast<Int*>(&len)); ++wraps; }
}

// ─── compact DP ───────────────────────────────────────────────────────────────
using fp_t = uint64_t;
inline fp_t make_fp(const Point &P){
    return splitmix64(IntLow64(P.x) ^ uint64_t(!P.y.IsEven()));
}

static std::vector<fp_t>       fp_tbl;
static std::vector<Scalar256>  idx_tbl;
static std::unique_ptr<std::atomic<uint8_t>[]> used_tbl;
static uint32_t  dp_cap=0;
static std::atomic<uint64_t> dpDone{0};
static uint64_t  dpTarget=0;

inline bool sameScalar(const Scalar256 &a,const Scalar256 &b){
    return std::memcmp(&a,&b,sizeof(Scalar256))==0;
}

inline bool dp_insert_unique(fp_t fp,const Int &idx){
    Int modIdx; intCopy(modIdx, idx); modIdx.Mod(&ORDER_N);
    Scalar256 key; intToScalar(modIdx,key);
    uint32_t h = uint32_t(fp) % dp_cap;
    for(;;){
        uint8_t st = used_tbl[h].load(std::memory_order_acquire);
        if(st==2){
            if(fp_tbl[h]==fp && sameScalar(idx_tbl[h],key)) return false;
        }else if(st==0){
            uint8_t exp=0;
            if(used_tbl[h].compare_exchange_strong(exp,1,std::memory_order_acq_rel)){
                fp_tbl[h]=fp; idx_tbl[h]=key;
                used_tbl[h].store(2,std::memory_order_release);
                dpDone.fetch_add(1,std::memory_order_relaxed);
                return true;
            }
        }
        if(++h==dp_cap) h=0;
    }
}
inline bool dp_find(fp_t fp,Int &idx){
    uint32_t h=uint32_t(fp)%dp_cap;
    while(used_tbl[h].load(std::memory_order_acquire)==2){
        if(fp_tbl[h]==fp){ scalarToInt(idx_tbl[h],idx); return true; }
        if(++h==dp_cap) h=0;
    }
    return false;
}

// ─── globals ──────────────────────────────────────────────────────────────────
static simd_bloom::SimdBlockFilterFixed<> *bloom=nullptr;
static std::atomic<uint64_t> hops{0};
static std::atomic<uint64_t> restarts{0};
static std::atomic<bool>     solved{false};
static Int                   privFound;
static std::vector<Point>    jumps;
static std::atomic<unsigned> found_tid{0};
static std::atomic<uint64_t> winner_wraps{0};

// solving fix
static std::once_flag                record_flag;
static std::chrono::steady_clock::time_point t_end;

// ─── batch-EC-add ─────────────────────────────────────────────────────────────
template<unsigned N>
static inline void batchAdd(Point *base,Point *plus){
    std::array<Int,N> dX;
    for(unsigned i=0;i<N;++i) dX[i].ModSub(&plus[i].x,&base[i].x);
    static thread_local IntGroup grp(N); grp.Set(dX.data()); grp.ModInv();
    for(unsigned i=0;i<N;++i){
        Int dY; dY.ModSub(&plus[i].y,&base[i].y);
        Int k;  k.ModMulK1(&dY,&dX[i]);
        Int k2; k2.ModSquareK1(&k);
        Int xN(base[i].x); xN.ModNeg(); xN.ModAdd(&k2); xN.ModSub(&plus[i].x);
        Int dx(base[i].x); dx.ModSub(&xN); dx.ModMulK1(&k);
        base[i].x=xN; base[i].y.ModNeg(); base[i].y.ModAdd(&dx);
    }
}

// ─── jump-table ───────────────────────────────────────────────────────────────
static void buildJumpTable(unsigned k){
    jumps.resize(k);
#pragma omp parallel for schedule(static)
    for(unsigned i=0;i<k;++i){
        Int e((uint64_t)1); e.ShiftL(int(i+1)); jumps[i]=mulP(e);
    }
}

// ─── Binary DP File Format ────────────────────────────────────────────────────
#pragma pack(push, 1)
struct DpItem {
    fp_t fp;
    uint8_t priv[32];
};
#pragma pack(pop)

static void saveDPBinary(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::cerr << "[ERROR] Cannot open " << filename << " for writing\n";
        return;
    }

    uint64_t count = 0;
    for (uint32_t h = 0; h < dp_cap; ++h) {
        if (used_tbl[h].load(std::memory_order_acquire) == 2) {
            DpItem item;
            item.fp = fp_tbl[h];
          
            Int priv;
            scalarToInt(idx_tbl[h], priv);
            priv.Get32Bytes(item.priv);
          
            file.write(reinterpret_cast<const char*>(&item), sizeof(DpItem));
            count++;
        }
    }

    std::cout << "Saved " << count << " DPs to " << filename
              << " (" << humanBytes(file.tellp()) << ")\n";
}

static bool loadDPBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[ERROR] Cannot open " << filename << " for reading\n";
        return false;
    }

    auto fileSize = file.tellg();
    file.seekg(0);
  
    if (fileSize % sizeof(DpItem) != 0) {
        std::cerr << "[ERROR] Invalid DP file size\n";
        return false;
    }

    const uint64_t count = fileSize / sizeof(DpItem);
    std::cout << "Loading " << count << " DPs from " << filename << "\n";

    DpItem item;
    uint64_t loaded = 0;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(DpItem))) {
        Int priv;
        priv.Set32Bytes(item.priv);
      
        if (dp_insert_unique(item.fp, priv)) {
            bloom->Add(uint32_t(item.fp));
            loaded++;
        }

        if (loaded % 1000000 == 0) {
            std::cout << "\rLoaded " << loaded << "/" << count << " DPs" << std::flush;
        }
    }

    std::cout << "\rLoaded " << loaded << " DPs (done)\n";
    return true;
}

// ─── Traps (Phase-1) ─────────────────────────────────────────────────────────
static constexpr unsigned K_DP=512;
static void buildDP_segment(const RangeSeg &seg,uint64_t target,
                            unsigned k,unsigned dp_bits,uint64_t seed)
{
    const uint64_t mask=(1ULL<<dp_bits)-1;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> rd;
    std::array<Int,   K_DP> dist;
    std::array<uint64_t,K_DP> wraps{};
    std::array<Point, K_DP> cur, stepPts;

    auto rndMod=[&](Int &o){
        o.SetInt32(0); int parts=(bitlen(seg.length)+63)/64;
        for(int p=0;p<parts;++p){
            Int t((uint64_t)rd(rng)); t.ShiftL(p*64); o.Add(&t);
        }
        o.Mod(const_cast<Int*>(&seg.length));
    };

    for(unsigned i=0;i<K_DP;++i){
        rndMod(dist[i]);
        Int a(seg.start); a.Add(&dist[i]);
        cur[i]=mulP(a);
    }

    uint64_t made=0;
    while(made<target){
        for(unsigned i=0;i<K_DP;++i){
            uint64_t h=splitmix64(IntLow64(cur[i].x))%k;
            Int step((uint64_t)1); step.ShiftL(int(h+1));

            if((IntLow64(cur[i].x)&mask)==0){
                fp_t fp=make_fp(cur[i]);

                Int scalar(seg.length);
                Int w((uint64_t)wraps[i]); scalar.Mult(&w);
                scalar.Add(const_cast<Int*>(&dist[i]));
                scalar.Add(const_cast<Int*>(&seg.start));
                scalar.Mod(&ORDER_N);

                if(dp_insert_unique(fp,scalar)){
                    bloom->Add(uint32_t(fp));
                    if(++made==target) break;
                }
            }
            stepPts[i]=jumps[h];
            addWrapCnt(dist[i],step,seg.length,wraps[i]);
        }
        batchAdd<K_DP>(cur.data(),stepPts.data());
    }
}

// ─── Phase-2: wild kangaroos ─────────────────────────────────────────────────
static constexpr unsigned K=512, BUF=512;
static void worker(uint32_t tid,const RangeSeg &seg,const Point &pub,
                   unsigned k,unsigned dp_bits)
{
    struct LoopDet{ uint64_t next,cnt,sig;
        inline void reset(uint64_t s) noexcept{ next=1024; cnt=0; sig=s; } };

    const uint64_t mask=(1ULL<<dp_bits)-1;
    std::mt19937_64 rng(splitmix64(0xDEADBEEF*tid));
    std::uniform_int_distribution<uint64_t> rd;

    std::array<Int,   K> dist;
    std::array<uint64_t,K> wraps{};
    std::array<Point, K> cur, stepPts;
    std::array<LoopDet,K> loop;

    auto rndMod=[&](Int &o){
        o.SetInt32(0); int parts=(bitlen(seg.length)+63)/64;
        for(int p=0;p<parts;++p){
            Int t((uint64_t)rd(rng)); t.ShiftL(p*64); o.Add(&t);
        }
        o.Mod(const_cast<Int*>(&seg.length));
    };
    for(unsigned i=0;i<K;++i){
        rndMod(dist[i]); cur[i]=addP(pub,mulP(dist[i]));
        uint64_t sig=splitmix64(IntLow64(cur[i].x)^uint64_t(!cur[i].y.IsEven()));
        loop[i].reset(sig);
    }

    const uint64_t FLUSH=1ULL<<18;
    uint64_t local=0; std::array<fp_t,BUF> fpB; std::array<unsigned,BUF> idB; unsigned cnt=0;

    while(!solved.load()){
        for(unsigned i=0;i<K;++i){
            if(solved.load()) return;

            uint64_t x64=IntLow64(cur[i].x);
            uint64_t h  =splitmix64(x64)%k;

            // Brent loop-detector ──────────────────────────────
            LoopDet &ld=loop[i];
            if(++ld.cnt==ld.next){
                uint64_t sig=splitmix64(x64^uint64_t(!cur[i].y.IsEven()));
                if(sig==ld.sig){
                    rndMod(dist[i]);
                    cur[i]=addP(pub,mulP(dist[i]));
                    wraps[i]=0;
                    ld.reset(sig);
                    restarts.fetch_add(1,std::memory_order_relaxed);
                    continue;
                }
                ld.sig=sig; ld.next<<=1;
            }

            stepPts[i]=jumps[h];
            Int step((uint64_t)1); step.ShiftL(int(h+1));
            addWrapCnt(dist[i],step,seg.length,wraps[i]);
            ++local;
        }
        batchAdd<K>(cur.data(),stepPts.data());
        if(local>=FLUSH){ hops.fetch_add(local); local=0; }

        if(solved.load()) return;

        for(unsigned i=0;i<K;++i){
            if(solved.load()) return;
            if((IntLow64(cur[i].x)&mask)!=0) continue;
            fp_t fp=make_fp(cur[i]);
            fpB[cnt]=fp; idB[cnt]=i;
            if(++cnt==BUF){
                for(unsigned j=0;j<BUF;++j){
                    if(!bloom->Find(uint32_t(fpB[j]))) continue;
                    Int trap; if(!dp_find(fpB[j],trap)) continue;

                    Int dw(seg.length);
                    Int w((uint64_t)wraps[idB[j]]); dw.Mult(&w);
                    dw.Add(const_cast<Int*>(&dist[idB[j]]));
                    dw.Mod(&ORDER_N);

                    Int priv; intCopy(priv,trap);
                    priv.Sub(&dw); priv.Mod(&ORDER_N);

                    Point tst=mulP(priv);
                    if(tst.x.IsEqual(&const_cast<Int&>(pub.x)) &&
                       tst.y.IsEqual(&const_cast<Int&>(pub.y))){
                        std::call_once(record_flag,[&]{ t_end=std::chrono::steady_clock::now(); });
                        intCopy(privFound,priv);
                        found_tid.store(tid,std::memory_order_relaxed);
                        winner_wraps.store(wraps[idB[j]],std::memory_order_relaxed);
                        solved.store(true); return;
                    }
                }
                cnt=0;
            }
        }
    }
    if(local) hops.fetch_add(local);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc,char** argv){
    P_PRIME=hexToInt(P_HEX); ORDER_N=hexToInt(N_HEX); secp.Init();

    Int A,B; uint64_t traps=0; unsigned dp_bits=12;
    const double bloomFactor=2.0, MAX_LOAD=0.75;
    Point pub; bool saveDP=false; size_t ramLimitGB=16;
    unsigned k_user=0;
    bool loadDP=false;
    std::string dpFile;

    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--range"){
            std::string s=argv[++i]; auto p=s.find(':');
            A=decToInt(s.substr(0,p)); B=decToInt(s.substr(p+1));
        }else if(a=="--dp_point") traps=std::strtoull(argv[++i],nullptr,10);
        else if(a=="--dp_bits")   dp_bits=std::stoul(argv[++i]);
        else if(a=="--ram"||a=="--ram-limit") ramLimitGB=std::stoull(argv[++i]);
        else if(a=="--pubkey"){
            std::string h=argv[++i]; if(h.rfind("0x",0)==0) h.erase(0,2);
            char pc=h[1]; Int x=hexToInt(h.substr(2));
            pub.x=x; pub.y=secp.GetY(x,pc=='2');
        }else if(a=="-s"||a=="--save-dp") saveDP=true;
        else if(a=="--k") k_user=std::stoul(argv[++i]);
        else if(a=="--load-dp") {
            loadDP = true;
            dpFile = argv[++i];
        }
        else{ std::cerr<<"Unknown option "<<a<<'\n'; return 1; }
    }
    if(A.IsZero()||B.IsZero()){ std::cerr<<"range not set\n"; return 1; }

    Int range(B); range.Sub(&A);
    unsigned Lbits=bitlen(range);

    if(!traps && !loadDP){
        traps=(Lbits>=52)?(1ULL<<(Lbits/2)):
              uint64_t(std::ceil(range.ToDouble()/std::sqrt(range.ToDouble())));
        std::cout<<"[auto] dp_point = "<<traps<<'\n';
    }
    unsigned k = k_user ? k_user : std::max(1u, Lbits/2);
    buildJumpTable(k);

    // ---------- RAM -------------------------------------------------------
    uint32_t need  = uint32_t(std::ceil(double(traps)/MAX_LOAD));
    dp_cap         = need;
    double load    = double(traps)/dp_cap;

    const size_t slotBytes = sizeof(fp_t)+sizeof(Scalar256)+1;
    size_t dpBytes   = size_t(dp_cap)*slotBytes;
    size_t bloomBytes= size_t(traps*bloomFactor);
    size_t totalBytes= dpBytes+bloomBytes;

    std::cout<<"\n=========== Phase-0: RAM summary ===========\n";
    std::cout<<"DP table : "<<humanBytes(dpBytes)
             <<"  ( "<<traps<<" / "<<dp_cap<<" slots, load "
             <<std::fixed<<std::setprecision(2)<<load*100<<"% )\n";
    std::cout<<"Bloom    : "<<humanBytes(bloomBytes)<<"\n";
    std::cout<<"--------------------------------------------\n";
    std::cout<<"Total    : "<<humanBytes(totalBytes)<<'\n';

    if(totalBytes>ramLimitGB*(size_t(1)<<30)){
        std::cerr<<"Error: need "<<humanBytes(totalBytes)
                 <<" (> "<<ramLimitGB<<" GiB)\n"; return 1;
    }

    // ---------- allocate -----------------------------------------------------
    fp_tbl.assign(dp_cap,0); idx_tbl.assign(dp_cap,Scalar256{0,0,0,0});
    used_tbl.reset(new std::atomic<uint8_t>[dp_cap]);
    for(uint32_t h=0;h<dp_cap;++h) used_tbl[h].store(0);
    dpTarget=traps; bloom=new simd_bloom::SimdBlockFilterFixed<>(bloomBytes);

    unsigned th=std::max(1u,std::thread::hardware_concurrency());
    auto segments=splitRange(A,range,th);
    uint64_t per=(traps+th-1)/th;

    std::cout<<"\n====== Phase-1: Building/Loading traps =====\n";
    if (loadDP) {
        if (!loadDPBinary(dpFile)) {
            return 1;
        }
    } else {
        std::thread progress([&]{
            while(dpDone.load()<dpTarget){
                std::cout<<"\rUnique traps: "<<dpDone<<'/'<<dpTarget<<std::flush;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::cout<<"\rUnique traps: "<<dpTarget<<'/'<<dpTarget<<" (done)\n";
        });
#pragma omp parallel for schedule(static)
        for(unsigned t=0;t<th;++t)
            buildDP_segment(segments[t],per,k,dp_bits,splitmix64(0xABCDEF12345678ULL^t));
        progress.join();

        if (saveDP) {
            saveDPBinary("DP.bin");
        }
    }

    // ─── Phase-2: Kangaroos ─────────────────────────────────────────────────
    std::cout<<"\n=========== Phase-2: Kangaroos =============\n";
    auto t0 = std::chrono::steady_clock::now(); uint64_t last=0;
    std::thread pool([&]{
#pragma omp parallel for num_threads(th) schedule(static)
        for(unsigned id=0;id<th;++id)
            worker(id,segments[id],pub,k,dp_bits);
    });

    while(!solved.load()){
        std::this_thread::sleep_for(std::chrono::seconds(5));
        uint64_t now=hops.load(), d=now-last; last=now;
        double disp=d/5.0; const char* u=" H/s";
        if(disp>1e6){ disp/=1e6; u=" MH/s"; }
        else if(disp>1e3){ disp/=1e3; u=" kH/s"; }
        auto dt=std::chrono::steady_clock::now()-t0;
        uint64_t s=std::chrono::duration_cast<std::chrono::seconds>(dt).count();
        std::cout<<"\rSpeed: "<<std::fixed<<std::setprecision(2)<<disp<<u
                 <<" | Hops: "<<now
                 <<" | Restart wild: "<<restarts.load()
                 <<" | Time: "<<s/3600<<':'<<((s/60)%60)<<':'<<s%60
                 <<std::flush;
    }
    pool.join();

    // ─── Phase-3: results -------------------------------------------
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t0);
    uint64_t total_ms = elapsed.count();
    uint64_t hours=total_ms/3600000, rem=total_ms%3600000;
    uint64_t minutes=rem/60000; rem%=60000;
    uint64_t seconds=rem/1000, millis=rem%1000;

    std::cout<<"\n\n============= Phase-3: Result ==============\n";
    std::cout<<"Private key : 0x"<<intHex(privFound,true)<<'\n';
    std::cout<<"Found by thr: "<<found_tid.load()<<'\n';
    uint64_t wcnt = winner_wraps.load();
    std::cout<<"Wild wraps  : "<<wcnt<<(wcnt?"  [wrapped]\n":"  [no wrap]\n");
    std::cout<<"Wild restart: "<<restarts.load()<<'\n';
    std::cout<<"Total time  : "
             <<std::setfill('0')<<std::setw(2)<<hours<<':'
             <<std::setw(2)<<minutes<<':'<<std::setw(2)<<seconds<<'.'
             <<std::setw(3)<<millis<<'\n';

    {   std::ofstream fout("FOUND.txt",std::ios::trunc);
        if(!fout) std::cerr<<"[WARN] cannot open FOUND.txt\n";
        else{ fout<<"0x"<<intHex(privFound,true)<<'\n';
              std::cout<<"Private key : saved to FOUND.txt\n"; }
    }

    delete bloom;
    return 0;
}