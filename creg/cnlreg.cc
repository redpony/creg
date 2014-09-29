#include <cstdlib>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#if HAVE_NEW_CXX
# include <unordered_map>
# include <unordered_set>
#else
# include <tr1/unordered_map>
# include <tr1/unordered_set>
namespace std { using std::tr1::unordered_map; using std::tr1::unordered_set; }
#endif

#include <signal.h>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "json_feature_map_lexer.h"
#include "fdict.h"
#include "feature_map.h"
#include "prob.h"
#include "filelib.h"

#include <adept.h>
#include <Eigen/Eigen>

using adept::adouble;

#define INPUT_DIM 20
template <typename F> using FVector = Eigen::Matrix<F,INPUT_DIM,1>;

using namespace std;
namespace po = boost::program_options;

volatile bool* requested_stop = NULL;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("x,x", po::value<vector<string> >(), "Files containing training instance features")
        ("y,y", po::value<string>(), "File containing training instance responses (if unspecified, do prediction only)")
        ("tx", po::value<vector<string> >(), "Files containing training instance features")
        ("ty", po::value<string>(), "File containing training instance responses (if unspecified, do prediction only)")
        ("z,z", po::value<string>(), "Write learned weights to this file (optional)")
        ("l1",po::value<double>()->default_value(0.0), "l_1 regularization strength")
        ("l2",po::value<double>()->default_value(1e-10), "l_2 regularization strength")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

enum RegressionType { kLINEAR, kLOGISTIC };

struct TrainingInstance {
  FrozenFeatureMap x;
  union {
    unsigned label;  // for categorical & ordinal predictions
    float value;     // for continuous predictions
  } y;
};

struct ReaderHelper {
  explicit ReaderHelper(vector<TrainingInstance>* xyp,
                        bool h,
                        vector<string>* iids) : xy_pairs(xyp), lc(), flag(), has_labels(h), ids(iids), merge() {}
  unordered_map<string, unsigned> id2ind;
  FeatureMapStorage* fms;
  vector<TrainingInstance>* xy_pairs;
  int lc;
  bool flag;
  bool has_labels;
  vector<string>* ids;
  vector<bool> merged;
  bool merge;
};

void ReaderCB(const string& id,
              const std::pair<int,float>* begin,
              const std::pair<int,float>* end,
              void* extra) {
  ReaderHelper& rh = *reinterpret_cast<ReaderHelper*>(extra);
  ++rh.lc;
  if (rh.lc % 1000  == 0) { cerr << '.'; rh.flag = true; }
  if (rh.lc % 50000 == 0) { cerr << " [" << rh.lc << "]\n"; rh.flag = false; }
  if (rh.ids && !rh.merge) rh.ids->push_back(id);
  if (rh.has_labels) {
    const unordered_map<string, unsigned>::iterator it = rh.id2ind.find(id);
    if (it == rh.id2ind.end()) {
      cerr << "\nUnlabeled example in line " << rh.lc << " (key=" << id << ')' << endl;
      abort();
    } else {
      if (rh.merge) {
        rh.merged[it->second - 1] = true;
        const FrozenFeatureMap& prev = (*rh.xy_pairs)[it->second - 1].x;
        (*rh.xy_pairs)[it->second - 1].x = rh.fms->AddFeatureMap(begin,end,prev);
      } else {
        (*rh.xy_pairs)[it->second - 1].x = rh.fms->AddFeatureMap(begin,end);
      }
    }
  } else {
    TrainingInstance x_no_y;
    if (rh.merge) {
      if (!rh.ids) {
        cerr << "Missing IDs\n";
        abort();
      }
      unsigned& ind = rh.id2ind[id];
      rh.merged[ind - 1] = true;
      const FrozenFeatureMap& prev = (*rh.xy_pairs)[ind - 1].x;
      (*rh.xy_pairs)[ind - 1].x = rh.fms->AddFeatureMap(begin,end,prev);
    } else {
      x_no_y.x = rh.fms->AddFeatureMap(begin,end);
      unsigned& ind = rh.id2ind[id];
      assert(ind == 0);
      rh.xy_pairs->push_back(x_no_y);
      ind = rh.xy_pairs->size();
    }
  }
}

void ReadLabeledInstances(const vector<string>& ffeats,
                          const string& fresp,
                          const RegressionType resptype,
                          FeatureMapStorage* fms,
                          vector<TrainingInstance>* xy_pairs,
                          vector<string>* labels,
                          vector<string>* instance_ids = NULL) {
  bool flag = false;
  xy_pairs->clear();
  int lc = 0;
  ReaderHelper rh(xy_pairs, fresp.size() > 0, instance_ids);
  rh.merge = false;
  unordered_map<string, unsigned> label2id;
  if (fresp.size() == 0) {
    cerr << "No gold standard responses provided to learn from!" << endl;
  } else {
    cerr << "Reading responses from " << fresp << " ..." << endl;
    ReadFile fr(fresp);
    for (unsigned i = 0; i < labels->size(); ++i)
      label2id[(*labels)[i]] = i;
    istream& in = *fr.stream();
    string line;
    while(getline(in, line)) {
      ++lc;
      if (lc % 1000 == 0) { cerr << '.'; flag = true; }
      if (lc % 40000 == 0) { cerr << " [" << lc << "]\n"; flag = false; }
      if (line.size() == 0) continue;
      if (line[0] == '#') {
        if (line.size() > 1 && line[1] == '#') {
          if (lc != 1) {
            cerr << "[WARNING] Line " << lc << " appears to be label declaration ... ignoring\n";
          } else {
            istringstream is(line);
            string label;
            is >> label;
            while(is >> label) {
              unordered_map<string, unsigned>::iterator it = label2id.find(label);
              if (it == label2id.end()) {
                it = label2id.insert(make_pair(label, labels->size())).first;
                labels->push_back(label);
              }
            }
          }
        }
        continue;
      }
      unsigned p = 0;
      while (p < line.size() && line[p] != ' ' && line[p] != '\t') { ++p; }
      unsigned& ind = rh.id2ind[line.substr(0, p)];
      if (ind != 0) { cerr << "ID " << line.substr(0, p) << " duplicated in line " << lc << endl; abort(); }
      while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) { ++p; }
      assert(p < line.size());
      xy_pairs->push_back(TrainingInstance());
      ind = xy_pairs->size();
      switch (resptype) {
        case kLINEAR:
          xy_pairs->back().y.value = strtof(&line[p], 0);
          break;
        case kLOGISTIC:
          {
            unordered_map<string, unsigned>::iterator it = label2id.find(line.substr(p));
            if (it == label2id.end()) {
              const string label = line.substr(p);
              it = label2id.insert(make_pair(label, labels->size())).first;
              labels->push_back(label);
            }
            xy_pairs->back().y.label = it->second;  // label id
          }
          break;
      }
    }
    if (flag) cerr << endl;
    if (resptype == kLOGISTIC) {
      cerr << "LABELS:";
      for (unsigned j = 0; j < labels->size(); ++j)
        cerr << " " << (*labels)[j];
      cerr << endl;
    }
  }
  FeatureMapStorage* pfms = NULL;
  FeatureMapStorage* tfms = NULL;
  for (unsigned i = 0; i < ffeats.size(); ++i) {
    if (i == ffeats.size() - 1) {
      pfms = tfms;
      rh.fms = fms;
    } else {
      delete pfms;
      pfms = tfms;
      tfms = new FeatureMapStorage;
      rh.fms = tfms;
    }
    if (pfms) {
      rh.merge = true;
      rh.merged.clear();
      rh.merged.resize(rh.xy_pairs->size(), false);
    }
    const string& ffeat = ffeats[i];
    cerr << "Reading features from " << ffeat << " ..." << endl;
    ReadFile ff(ffeat);
    JSONFeatureMapLexer::ReadRules(ff.stream(), ReaderCB, &rh);
    if (pfms) {
      for (unsigned j = 0; j < rh.xy_pairs->size(); ++j)
        if (!rh.merged[j])
          (*rh.xy_pairs)[j].x = rh.fms->AddFeatureMap(NULL, NULL, (*rh.xy_pairs)[j].x);
    }
    if (rh.flag) cerr << endl;
  }
  delete pfms;
}

void signal_callback_handler(int /* signum */) {
  if (!requested_stop || *requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early.\n";
  *requested_stop = true;
}

template <typename F>
struct Model {
  Model(unsigned feats, unsigned labels) :
      input_reps(feats, FVector<F>::Zero()), output_reps(labels, FVector<F>::Zero()) {}

  // number of parameters
  size_t size() const {
    return input_reps.size() * input_reps[0].size() + output_reps.size() * output_reps[0].size();
  }

  void Randomize() {
    for (auto& iv: input_reps)
      iv = FVector<F>::Random() / 5.;
    for (auto& ov: output_reps)
      ov = FVector<F>::Random() / 2.;
  }

  template <typename T>
  void copyfrom(const Model<T>& o) {
    for (unsigned i = 0; i < input_reps.size(); ++i)
      for (unsigned j = 0; j < INPUT_DIM; ++j)
        input_reps[i](j,0) = o.input_reps[i](j,0);
    for (unsigned i = 0; i < output_reps.size(); ++i)
      for (unsigned j = 0; j < INPUT_DIM; ++j)
        output_reps[i](j,0) = o.output_reps[i](j,0);
  }

  void update(const Model<adouble>& g, Model<double>& h) {
    double eta = 0.1;
    for (unsigned i = 0; i < input_reps.size(); ++i)
      for (unsigned j = 0; j < INPUT_DIM; ++j) {
        double d = g.input_reps[i](j,0).get_gradient();
        if (d) {
          double s = h.input_reps[i](j,0) += d * d;
          input_reps[i](j,0) -= eta * d / sqrt(s);
        }
      }
    for (unsigned i = 0; i < output_reps.size(); ++i)
      for (unsigned j = 0; j < INPUT_DIM; ++j) {
        double d = g.output_reps[i](j,0).get_gradient();
        if (d) {
          double s = h.output_reps[i](j,0) += d * d;
          output_reps[i](j,0) -= eta * d / sqrt(s);
        }
      }
  }

  inline static void nonlinearity(FVector<F>& v) {
    for (unsigned i = 0; i < v.size(); ++i)
      v(i,0) = tanh(v(i,0));
  }

  F log_prob(const FrozenFeatureMap& x, unsigned y) const {
    FVector<F> h = FVector<F>::Zero();
    for (auto& it : x)
      h += input_reps[it.first] * it.second;
    nonlinearity(h);
    F z = 0;
    F res = 0;
    unsigned c = 0;
    for (auto& ov : output_reps) {
      F s = h.dot(ov);
      if (c == y) res = s;
      z += exp(s);
      ++c;
    }
    return res - log(z);
  }

  unsigned predict(const FrozenFeatureMap& x) const {
    FVector<F> h = FVector<F>::Zero();
    for (auto& it : x)
      h += input_reps[it.first] * it.second;
    nonlinearity(h);
    unsigned c = 0;
    F cur_best = 0.0;
    unsigned cur_best_i = output_reps.size();
    for (auto& ov : output_reps) {
      F s = h.dot(ov);
      if (s > cur_best || cur_best_i == output_reps.size()) {
        cur_best = s; cur_best_i = c;
      }
      ++c;
    }
    return cur_best_i;
  }

  vector<FVector<F>> input_reps;
  vector<FVector<F>> output_reps;
};

int main(int argc, char** argv) {
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  string line;
  double l1 = conf["l1"].as<double>();
  double l2 = conf["l2"].as<double>();
  if (l1 < 0.0) {
    cerr << "L1 strength must be >= 0\n";
    return 1;
  }
  if (l2 < 0.0) {
    cerr << "L2 strength must be >= 0\n";
    return 2;
  }
  bool stop = false;
  requested_stop = &stop;
  RegressionType resptype = kLOGISTIC;
  if (conf.count("linear")) {
    resptype = kLINEAR;
  }
  if (!(conf.count("x") && conf.count("y"))) {
    cerr << "You must specify both training_features (-x) and training_responses (-y)!\n";
    return 1;
  }
  vector<string> labels; // only populated for non-continuous models
  vector<TrainingInstance> training, test;
  FeatureMapStorage fms;
  vector<string> xfile = conf["x"].as<vector<string> >();
  string yfile = conf["y"].as<string>();
  ReadLabeledInstances(xfile, yfile, resptype, &fms, &training, &labels);
  if (conf.count("tx") && conf.count("ty")) {
    vector<string> xfile = conf["tx"].as<vector<string> >();
    string yfile = conf["ty"].as<string>();
    ReadLabeledInstances(xfile, yfile, resptype, &fms, &test, &labels);
  }

  adept::Stack s;
  cerr << "         Number of features: " << FD::NumFeats() << endl;
  cerr << "Number of training examples: " << training.size() << endl;
  const unsigned p = FD::NumFeats();

  Model<double> dmodel(p,labels.size());
  dmodel.Randomize();
  Model<double> hmodel(p,labels.size());
  Model<adouble> amodel(p,labels.size());
  cout.precision(15);
  ostream* out = &cout;
  cerr << "Model size: " << dmodel.size() << " parameters\n";
  // set up signal handler to catch SIGINT
  signal(SIGINT, signal_callback_handler);

  if (conf.count("linear")) {  // linear regression
    cerr << "Not implemented!\n";
    return 1;
  } else {                     // logistic regression
    for (unsigned iter = 0; !stop && iter < 100000; ++iter) {
      amodel.copyfrom(dmodel);
      s.new_recording();
      adouble llh = 0;
      double right = 0;
      for (auto& xy: training) {
        adouble lp = amodel.log_prob(xy.x, xy.y.label);
        unsigned pred_y = dmodel.predict(xy.x);
        if (pred_y == xy.y.label) right++;
        //cerr << xy.y.label << " ||| " << lp << endl;
        llh -= lp;
      }
      llh.set_gradient(1.0);
      s.compute_adjoint();
      double hright = 0.0;
      double hllh = 0;
      unsigned lc = 0;
      for (auto& xy: test) {
        ++lc;
        double lp = dmodel.log_prob(xy.x, xy.y.label);
        unsigned pred_y = dmodel.predict(xy.x);
        if (pred_y == xy.y.label) hright++;
        else if (lc < 100) {
          cerr << "line " << lc << " which is in " << labels[xy.y.label] << " was confused for " << labels[pred_y] << endl;
        }
        //cerr << xy.y.label << " ||| " << lp << endl;
        hllh -= lp;
      }
      double perp = exp(llh.value() / training.size());
      double perpho = exp(hllh / test.size());
      cerr << "i=" << iter << "\tPerplexity: " << perp << "\ttrain-acc: " << (right / training.size());
      if (test.size()) cerr << "\td-perp: " << perpho << "\ttd-acc: " << (hright / test.size());
      cerr << endl;
      dmodel.update(amodel, hmodel);
    }
  }
  for (unsigned i = 0; i < labels.size(); ++i) {
    cerr << labels[i] << " " << dmodel.output_reps[i].transpose() << endl;
  }
  if (out && out != &cout)
    delete out;

  return 0;
}
