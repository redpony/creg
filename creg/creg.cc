#include <cstdlib>
#include <iostream>
#include <vector>
#include <tr1/unordered_map>
#include <limits>
#include <cmath>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "json_feature_map_lexer.h"
#include "fdict.h"
#include "feature_map.h"
#include "prob.h"
#include "filelib.h"
#include "liblbfgs/lbfgs++.h"

using namespace std;
using namespace std::tr1;
namespace po = boost::program_options;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("linear,n", "Linear regression (default is multiclass logistic regression)")
        ("ordinal,o", "Ordinal regression (proportional odds)")
        ("training_features,x", po::value<string>(), "File containing training instance features")
        ("training_responses,y", po::value<string>(), "File containing training instance responses")
        ("tx", po::value<string>(), "File containing test instance features (optional)")
        ("ty", po::value<string>(), "File containing test instance responses (optional)")
        ("write_test_predictions,p", "Write model prediction for each test instance")
        ("write_test_distribution,D", "Write posterior distribution of outputs for each test instance (categorical models only)")
        ("dont_write_weights,W", "Supress writing learned weights")
        ("multiclass_test_probability_threshold,P", po::value<double>()->default_value(0.0), "When evaluating a multiclass model, only compute the accuracy on instances where the predicted class posterior probability is > P")
        ("l1",po::value<double>()->default_value(0.0), "l_1 regularization strength")
        ("l2",po::value<double>()->default_value(1e-10), "l_2 regularization strength")
        ("temperature,T",po::value<double>()->default_value(0), "Temperature for entropy regularization (> 0 flattens, < 0 sharpens; = 0 no effect)")
        ("weights,w", po::value<string>(), "Initial weights file")
        ("epsilon,e", po::value<double>()->default_value(1e-4), "Epsilon for convergence test. Terminates when ||g|| < epsilon * max(1, ||w||)")
        ("delta,d", po::value<double>()->default_value(0), "Delta for convergence test. Terminates when (f' - f) / f < delta")
        ("memory_buffers,m",po::value<unsigned>()->default_value(40), "Number of memory buffers for LBFGS")
        ("test_features,t", po::value<string>(), "[deprecated option, use --tx] File containing test instance features")
        ("test_responses,s", po::value<string>(), "[deprecated option, use --ty] File containing test response features (ARKRegression format)")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help") || !conf->count("training_features") || !conf->count("training_responses")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

enum RegressionType { kLINEAR, kLOGISTIC, kORDINAL };

struct TrainingInstance {
  FrozenFeatureMap x;
  union {
    unsigned label;  // for categorical & ordinal predictions
    float value;     // for continuous predictions
  } y;
};

struct ReaderHelper {
  explicit ReaderHelper(FeatureMapStorage* pfms, vector<TrainingInstance>* xyp, bool h, vector<string>* iids) : fms(pfms), xy_pairs(xyp), lc(), flag(), has_labels(h), ids(iids) {}
  unordered_map<string, unsigned> id2ind;
  FeatureMapStorage* fms;
  vector<TrainingInstance>* xy_pairs;
  int lc;
  bool flag;
  bool has_labels;
  vector<string>* ids;
};

void ReaderCB(const string& id,
              const std::pair<int,float>* begin,
              const std::pair<int,float>* end,
              void* extra) {
  ReaderHelper& rh = *reinterpret_cast<ReaderHelper*>(extra);
  ++rh.lc;
  if (rh.lc % 1000  == 0) { cerr << '.'; rh.flag = true; }
  if (rh.lc % 50000 == 0) { cerr << " [" << rh.lc << "]\n"; rh.flag = false; }
  if (rh.ids) rh.ids->push_back(id);
  if (rh.has_labels) {
    const unordered_map<string, unsigned>::iterator it = rh.id2ind.find(id);
    if (it == rh.id2ind.end()) {
      cerr << "\nUnlabeled example in line " << rh.lc << " (key=" << id << ')' << endl;
      abort();
    } else {
      (*rh.xy_pairs)[it->second - 1].x = rh.fms->AddFeatureMap(begin,end);
    }
  } else {
    TrainingInstance x_no_y;
    x_no_y.x = rh.fms->AddFeatureMap(begin,end);
    rh.xy_pairs->push_back(x_no_y);
  }
}

void ReadLabeledInstances(const string& ffeats,
                          const string& fresp,
                          const RegressionType resptype,
                          FeatureMapStorage* fms,
                          vector<TrainingInstance>* xy_pairs,
                          vector<string>* labels,
                          vector<string>* instance_ids = NULL) {
  bool flag = false;
  xy_pairs->clear();
  int lc = 0;
  ReaderHelper rh(fms, xy_pairs, fresp.size() > 0, instance_ids);
  unordered_map<string, unsigned> label2id;
  if (fresp.size() == 0) {
    cerr << "No gold standard responses provided for " << ffeats << endl;
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
      if (line[0] == '#') continue;
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
        case kORDINAL:
          {
            // TODO allow labels not to be consecutive and start from 0 
            // requires label re-indexing
            const unsigned label = strtol(&line[p], 0, 10);
            xy_pairs->back().y.label = label;
            if (label >= labels->size()) {
              labels->resize(label + 1);
            }
            (*labels)[label] = line.substr(p);
          }
          break;
      }
    }
    if (flag) cerr << endl;
    if (resptype == kLOGISTIC || resptype == kORDINAL) {
      cerr << "LABELS:";
      for (unsigned j = 0; j < labels->size(); ++j)
        cerr << " " << (*labels)[j];
      cerr << endl;
    }
  }
  cerr << "Reading features from " << ffeats << " ..." << endl;
  ReadFile ff(ffeats);
  JSONFeatureMapLexer::ReadRules(ff.stream(), ReaderCB, &rh);
  if (rh.flag) cerr << endl;
}

// helper base class (not polymorphic- just a container and some helper functions) for loss functions
// real loss functions should implement double operator()(const vector<double>& x, double* g),
// which should evaluate f(x) and g = f'(x)
struct BaseLoss {
  // dimp1 = number of categorial outputs possible for logistic regression
  // for linear regression, it should be 1 more than the dimension of the response variable
  BaseLoss(
      const vector<TrainingInstance>& tr,
      unsigned dimp1,
      unsigned numfeats,
      unsigned ll2) : training(tr), K(dimp1), p(numfeats), l2(ll2) {}

  // w.x (bias excluded)
  template <class FeatureMapType>
  double DotProduct(const FeatureMapType& fx,
                    const vector<double>& w) const {
    const unsigned km1 = K - 1;
    double dotproduct = 0;
    for (typename FeatureMapType::const_iterator it = fx.begin(); it != fx.end(); ++it)
      dotproduct += w[it->first + km1] * it->second;
    return dotproduct;
  }

  double ApplyRegularizationTerms(const vector<double>& weights,
                                  double* g) const {
    double reg = 0;
    for (size_t i = K - 1; i < weights.size(); ++i) {
      const double& w_i = weights[i];
      reg += l2 * w_i * w_i;
      g[i] += 2 * l2 * w_i;
    }
    return reg;
  }

  template <class FeatureMapType>
  void GradAdd(const FeatureMapType& fx,
               const unsigned y,
               const double scale,
               double* acc) const {
    acc[y] += scale; // class bias
    for (typename FeatureMapType::const_iterator it = fx.begin();
         it != fx.end(); ++it)
      acc[it->first + y * p + K - 1] += it->second * scale;
  }

  const vector<TrainingInstance>& training;
  const unsigned K, p;
  const double l2;
};

struct UnivariateSquaredLoss : public BaseLoss {

  // weight vector layout for p features
  //   w[0] = bias weight
  //   w[1 : p] = feature weights

  UnivariateSquaredLoss(
          const vector<TrainingInstance>& tr,
          unsigned numfeats,
          const double l2) : BaseLoss(tr, 2, numfeats, l2) {}

  template <class FeatureMapType>
  double Predict(const FeatureMapType& fx, const vector<double>& w) const {
    return DotProduct(fx, w) + w[0];
  }

  // evaluate squared loss and gradient
  double operator()(const vector<double>& x, double* g) const {
    fill(g, g + x.size(), 0.0);
    double cll = 0;
    for (unsigned i = 0; i < training.size(); ++i) {
      const FrozenFeatureMap& fmapx = training[i].x;
      const double refy = training[i].y.value;
      const double predy = Predict(fmapx, x);
      const double diff = predy - refy;
      cll += diff * diff;
      GradAdd(fmapx, 0, 2 * diff, g);
    }
    const double reg = ApplyRegularizationTerms(x, g);
    return cll + reg;
  }

  // return RMSE
  double Evaluate(const vector<TrainingInstance>& test,
                  const vector<double>& w,
                  vector<float>* preds = NULL) const {
    vector<double> dotprods(1);  // K-1 degrees of freedom
    double mse = 0;
    if (preds) preds->resize(test.size());
    for (unsigned i = 0; i < test.size(); ++i) {
      const double predy = Predict(test[i].x, w);
      if (preds) (*preds)[i] = predy;
      const double refy = test[i].y.value;
      const double diff = predy - refy;
      //cerr << "line=" << (i+1) << " true=" << refy << " pred=" << predy << endl;
      mse += diff * diff;
    }
    mse /= test.size();
    return sqrt(mse);
  }
};

// predictions made by multiclass logistic regression for some
// input x
//   y_hat is the 1-best prediction
//   posterior[y] is p(y_hat | x)
struct MulticlassPrediction {
  unsigned y_hat;
  vector<double> posterior;
};

struct MulticlassLogLoss : public BaseLoss {

  // weight vector layout for K classes, with p features
  //   w[0 : K-2] = bias weights
  //   w[y*p + K-1 : y*p + K + p - 2] = feature weights for y^th class

  MulticlassLogLoss(
          const vector<TrainingInstance>& tr,
          unsigned k,
          unsigned numfeats,
          const double l2,
          const double t = 0.0) : BaseLoss(tr, k, numfeats, l2), T(t) {}

  //   g = E[ F(x,y) * log p(y|x) ] + H(y | x) * E[ F(x,y) ]
  //   note: g will be scaled by T
  double Entropy(const prob_t& z,
                 const vector<double>& dotprods,
                 const FrozenFeatureMap& fmapx,
                 double* g) const {
    const double log_z = log(z);
    double entropy = log_z * exp(-log_z);   // class K dotprod = 0
    map<unsigned, double> ef;
    for (unsigned j = 0; j < dotprods.size(); ++j) {
      const double log_prob = dotprods[j] - log_z;
      const double prob = exp(log_prob);
      const double e_logprob = prob * log_prob;
      entropy -= e_logprob;
      GradAdd(fmapx, j, T * e_logprob, g);
    }
    for (unsigned j = 0; j < dotprods.size(); ++j) {
      const double log_prob = dotprods[j] - log_z;
      const double prob = exp(log_prob);
      GradAdd(fmapx, j, T * prob * entropy, g);
    }

    return entropy;
  }

  // evaluate log loss and gradient
  double operator()(const vector<double>& x, double* g) const {
    fill(g, g + x.size(), 0.0);
    vector<double> dotprods(K - 1);  // K-1 degrees of freedom
    vector<prob_t> probs(K);
    double cll = 0;
    double entropy = 0;  // only computed if T != 0
    for (unsigned i = 0; i < training.size(); ++i) {
      const FrozenFeatureMap& fmapx = training[i].x;
      const unsigned refy = training[i].y.label;
      ComputeDotProducts(fmapx, x, &dotprods);
      prob_t z;
      for (unsigned j = 0; j < dotprods.size(); ++j)
        z += (probs[j] = prob_t(dotprods[j], init_lnx()));
      z += (probs.back() = prob_t::One());
      for (unsigned y = 0; y < probs.size(); ++y) {
        probs[y] /= z;
        //cerr << "  p(y=" << y << ")=" << probs[y].as_float() << "\tz=" << z << endl;
      }
      cll -= log(probs[refy]);  // log p(y | x)

      for (unsigned y = 0; y < dotprods.size(); ++y) {
        double scale = probs[y].as_float();
        if (y == refy) { scale -= 1.0; }
        GradAdd(fmapx, y, scale, g);
      }
      if (T) entropy += Entropy(z, dotprods, fmapx, g);
    }
    double reg = ApplyRegularizationTerms(x, g);
    return cll + reg - T * entropy;
  }

  template <class FeatureMapType>
  pair<unsigned, double> Predict(const FeatureMapType& fx,
                                 const vector<double>& w,
                                 MulticlassPrediction* pred = NULL) const {
    vector<double> dotprods(K - 1);  // K-1 degrees of freedom
    if (pred) pred->posterior.resize(K);
    ComputeDotProducts(fx, w, &dotprods);
    prob_t z = prob_t::One();  // exp(0) for k^th class
    for (unsigned j = 0; j < dotprods.size(); ++j)
      z += prob_t(dotprods[j], init_lnx());
    const double log_z = log(z);
    double best = 0;
    unsigned besty = dotprods.size();
    for (unsigned y = 0; y < dotprods.size(); ++y) {
      if (dotprods[y] > best) { best = dotprods[y]; besty = y; }
      if (pred) pred->posterior[y] = exp(dotprods[y] - log_z);
    }
    if (pred) {
      pred->posterior.back() = exp(-log_z);
      pred->y_hat = besty;
    }
    return make_pair(besty, exp(best - log_z));
  }

  double Evaluate(const vector<TrainingInstance>& test,
                  const vector<double>& w,
                  double thresh_p,
		  vector<MulticlassPrediction>* preds = NULL) const {
    double correct = 0;
    unsigned examples = 0;
    if (preds) preds->resize(test.size());
    for (unsigned i = 0; i < test.size(); ++i) {
      MulticlassPrediction* ppred = NULL;
      if (preds) ppred = &(*preds)[i];
      const pair<unsigned, double> pred = Predict(test[i].x, w, ppred);
      const unsigned predy = pred.first;
      const unsigned refy = test[i].y.label;
      // cerr << "line=" << (i+1) << " true=" << refy << " pred=" << predy << "  p(y|x) = " << pred.second << endl;
      if (pred.second >= thresh_p) {
        ++examples;
        if (refy == predy) ++correct;
      }
    }
    return correct / examples;
  }

  template <class FeatureMapType>
  void ComputeDotProducts(const FeatureMapType& fx,  // feature vector of x
                          const vector<double>& w,         // full weight vector
                          vector<double>* pdotprods) const {
    vector<double>& dotprods = *pdotprods;
    const unsigned km1 = K - 1;
    dotprods.resize(km1);
    for (unsigned y = 0; y < km1; ++y)
      dotprods[y] = w[y];  // bias terms
    for (typename FeatureMapType::const_iterator it = fx.begin(); it != fx.end(); ++it) {
      const float fval = it->second;
      const unsigned fid = it->first;
      for (unsigned y = 0; y < km1; ++y)
        dotprods[y] += w[fid + y * p + km1] * fval;
    }
  }

  const double T; // temperature for entropy regularization
};

struct OrdinalLogLoss : public BaseLoss {

  // weight vector layout for K levels, with p features
  //   w[0 : K-2] = level biases
  //   w[K-1 : K + p - 2] = feature weights

  OrdinalLogLoss(
          const vector<TrainingInstance>& tr,
          unsigned k,
          unsigned numfeats,
          const double l2) : BaseLoss(tr, k, numfeats, l2) {}

  // evaluate log loss and gradient
  double operator()(const vector<double>& x, double* g) const {
    const unsigned km1 = K - 1;
    double cll = 0;
    fill(g, g + x.size(), 0.0);
    vector<double> u(K);
    u[0] = u[km1] = 0;
    for (unsigned k = 1; k < km1; k++)
      u[k] = 1 / (1 - exp(x[k] - x[k - 1]));
    for (unsigned i = 0; i < training.size(); ++i) {
      const FrozenFeatureMap& fmapx = training[i].x;
      const unsigned level = training[i].y.label;
      const double dotprod = DotProduct(fmapx, x);
      const double pj = LevelProb(dotprod, x, level);
      const double pjp1 = LevelProb(dotprod, x, level + 1);
      cll -= LogDeltaProb(dotprod, x, level);
      if (level > 0)
        g[level - 1] -= u[level] - 1 + pj;
      if (level < km1)
        g[level] -= - u[level] + pjp1;
      double scale = (1 - pj - pjp1);
      for (FrozenFeatureMap::const_iterator it = fmapx.begin();
          it != fmapx.end(); ++it)
        g[km1 + it->first] -= it->second * scale;
    }
    double reg = ApplyRegularizationTerms(x, g);
    return cll + reg;
  }

  template <class FeatureMapType>
  unsigned Predict(const FeatureMapType& fx, const vector<double>& w) const {
    const double dotprod = DotProduct(fx, w);
    for (unsigned k = 0; k < K; k++)
      if (dotprod < w[k]) return k;
    return K-1;
  }

  template <class FeatureMapType>
  unsigned Predict(const FeatureMapType& fx, const vector<double>& w, MulticlassPrediction* pred) const {
    const double dotprod = DotProduct(fx, w);
    pred->posterior.resize(K);
    double bestp = -1;
    unsigned besty = 0;
    for (unsigned k = 0; k < K; ++k) {
      double p = exp(LogDeltaProb(dotprod, w, k));
      pred->posterior[k] = p;
      if (p > bestp) { bestp = p; besty = k; }
    }
    pred->y_hat = besty;
    return besty;
  }

  double Evaluate(const vector<TrainingInstance>& test,
                  const vector<double>& w,
                  vector<MulticlassPrediction>* predictions = NULL) const {
    double correct = 0;
    if (predictions) predictions->resize(test.size());
    MulticlassPrediction dummy;
    for (unsigned i = 0; i < test.size(); ++i) {
      MulticlassPrediction* pred = predictions ? &(*predictions)[i] : &dummy;
      const unsigned predy = Predict(test[i].x, w, pred);
      const unsigned refy = test[i].y.label;
      if (refy == predy) correct++;
    }
    return correct / test.size();
  }

  double LevelProb(double dotprod, const vector<double>& w,
                   unsigned level) const { // p(y >= level+1)
    if (level == K) return 0; // p(y > K) = 0
    if (level == 0) return 1; // p(y >= 1) = 1
    return 1 / (1 + exp(w[level - 1] - dotprod));
  }

  double LogDeltaProb(double dotprod, const vector<double>& w,
                      unsigned level) const { // log p(y == level + 1)
        if (level == K-1) {
          prob_t zj = prob_t(dotprod, init_lnx()) + prob_t(w[K-2], init_lnx());
          return dotprod - log(zj);
        }
        if (level == 0) {
          prob_t zjp1 = prob_t(dotprod, init_lnx()) + prob_t(w[0], init_lnx());
          return w[0] - log(zjp1);
        }
        if (w[level] <= w[level - 1]) return -1e3;
        prob_t zj = prob_t(dotprod, init_lnx()) + prob_t(w[level - 1], init_lnx());
        prob_t zjp1 = prob_t(dotprod, init_lnx()) + prob_t(w[level], init_lnx());
        prob_t dalpha = prob_t(w[level], init_lnx()) - prob_t(w[level - 1], init_lnx());
        return (dotprod + log(dalpha) - log(zj) - log(zjp1));
      }

};

template <class LossFunction>
double LearnParameters(LossFunction& loss,
                       const double l1,
                       const unsigned l1_start,
                       const unsigned memory_buffers,
                       const double epsilon,
                       const double delta,
                       vector<double>* px) {
  LBFGS<LossFunction> lbfgs(px, loss, memory_buffers, l1, l1_start, epsilon, delta);
  lbfgs.MinimizeFunction();
  return 0;
}

int main(int argc, char** argv) {
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  string line;
  double l1 = conf["l1"].as<double>();
  double l2 = conf["l2"].as<double>();
  double temp = conf["temperature"].as<double>();
  const unsigned memory_buffers = conf["memory_buffers"].as<unsigned>();
  const double epsilon = conf["epsilon"].as<double>();
  const double delta = conf["delta"].as<double>();
  const double p_thresh = conf["multiclass_test_probability_threshold"].as<double>();
  if (l1 < 0.0) {
    cerr << "L1 strength must be >= 0\n";
    return 1;
  }
  if (l2 < 0.0) {
    cerr << "L2 strength must be >= 0\n";
    return 2;
  }
  if (p_thresh < 0.0 || p_thresh > 1.0) {
    cerr << "--multiclass_test_probability_threshold must be between 0 and 1\n";
    return 3;
  }

  RegressionType resptype = kLOGISTIC;
  if (conf.count("linear")) {
    if (conf.count("ordinal")) {
      cerr << "--ordinal and --linear are mutually exclusive\n";
      return 1;
    }
    resptype = kLINEAR;
  } else if (conf.count("ordinal")) {
    resptype = kORDINAL;
  }
  const string xfile = conf["training_features"].as<string>();
  const string yfile = conf["training_responses"].as<string>();
  vector<string> labels; // only populated for non-continuous models
  vector<TrainingInstance> training, test;
  FeatureMapStorage fms;
  ReadLabeledInstances(xfile, yfile, resptype, &fms, &training, &labels);

  if (conf.count("test_features")) {
    std::cerr << "THE OPTION --test_features IS DEPRECATED USE --tx AND --ty INSTEAD\n";
    const string txfile = conf["test_features"].as<string>();
    const string tyfile = conf["test_responses"].as<string>();
    ReadLabeledInstances(txfile, tyfile, resptype, &fms, &test, &labels);
  }

  bool test_labels = false;
  vector<string> test_ids;
  if (conf.count("tx")) {
    const string txfile = conf["tx"].as<string>();
    string tyfile;
    if (conf.count("ty")) tyfile = conf["ty"].as<string>();
    test_labels = tyfile.size();
    ReadLabeledInstances(txfile, tyfile, resptype, &fms, &test, &labels, &test_ids);
  }
  assert(test_ids.size() == test.size());

  if (conf.count("weights")) {
    cerr << "Initial weights are not implemented, please implement." << endl;
    // TODO read weights for categorical and continuous predictions
    // can't use normal cdec weight framework
    abort();
  }

  cerr << "         Number of features: " << FD::NumFeats() << endl;
  cerr << "Number of training examples: " << training.size() << endl;
  const unsigned p = FD::NumFeats();
  cout.precision(15);
  bool dont_write_weights = conf.count("dont_write_weights");
  bool write_dist = conf.count("write_test_distribution");
  bool write_pps = conf.count("write_test_predictions");

  if (conf.count("linear")) {  // linear regression
    vector<double> weights(1 + p, 0.0);
    cerr << "       Number of parameters: " << weights.size() << endl;
    UnivariateSquaredLoss loss(training, p, l2);
    LearnParameters(loss, l1, 1, memory_buffers, epsilon, delta, &weights);

    if (test.size()) {
      vector<float> preds;
      double rmse = loss.Evaluate(test, weights, &preds);
      if (test_labels) {
        cerr << "Held-out RMSE: " << rmse << endl;
      }
      if (!test_labels || write_pps) {
        for (unsigned i = 0; i < test.size(); ++i)
          cout << test_ids[i] << "\t" << preds[i] << endl;
      }
    }

    if (!dont_write_weights) {
      cout << p << "\t***CONTINUOUS***" << endl;
      cout << "***BIAS***\t" << weights[0] << endl;
      for (unsigned f = 0; f < p; ++f) {
        const double w = weights[1 + f];
        if (w)
          cout << FD::Convert(f) << "\t" << w << endl;
      }
    }
  } else if (conf.count("ordinal")) {
    const unsigned K = labels.size();
    const unsigned km1 = K - 1;
    vector<double> weights(p + km1, 0.0);
    for (unsigned k = 0; k < km1; k++)
      weights[k] = log(k+1) - log(K);
    OrdinalLogLoss loss(training, K, p, l2);
    LearnParameters(loss, l1, km1, memory_buffers, epsilon, delta, &weights);

    if (test.size()) {
      vector<MulticlassPrediction> predictions;
      double acc = loss.Evaluate(test, weights, &predictions);
      if (test_labels) {
        cerr << "Held-out accuracy: " << acc << endl;
      }
      if (!test_labels || write_dist || write_pps) {
        for (unsigned i = 0; i < test.size(); ++i) {
          cout << test_ids[i] << '\t' << predictions[i].y_hat;
          if (write_dist) {
            cout << "\t{";
            for (unsigned y = 0; y < K; ++y)
              cout << (y ? ", " : "") << '"' << labels[y] << "\": " << predictions[i].posterior[y];
            cout << '}';
          }
          cout << endl;
        }
      }
    }

    if (!dont_write_weights) {
      cout << p << "\t***ORDINAL***";
      for (unsigned y = 0; y < K; ++y)
        cout << '\t' << labels[y];
      cout << endl;
      for (unsigned y = 0; y < km1; ++y)
        cout << "y>=" << labels[y+1] << "\t" << weights[y] << endl;
      for (unsigned f = 0; f < p; ++f) {
        const double w = weights[km1 + f];
        if (w) cout << FD::Convert(f) << "\t" << w << endl;
      }
    }
  } else {                     // logistic regression
    vector<double> weights((1 + FD::NumFeats()) * (labels.size() - 1), 0.0);
    cerr << "       Number of parameters: " << weights.size() << endl;
    cerr << "           Number of labels: " << labels.size() << endl;
    const unsigned K = labels.size();
    const unsigned km1 = K - 1;
    MulticlassLogLoss loss(training, K, p, l2, temp);
    LearnParameters(loss, l1, km1, memory_buffers, epsilon, delta, &weights);

    if (test.size()) {
      vector<MulticlassPrediction> predictions;
      double acc = loss.Evaluate(test, weights, p_thresh, &predictions);
      if (test_labels) {
        cerr << "Held-out accuracy: " << acc << endl;
      }
      if (!test_labels || write_dist || write_pps) {
        for (unsigned i = 0; i < test.size(); ++i) {
          cout << test_ids[i] << '\t' << predictions[i].y_hat;
          if (write_dist) {
            cout << "\t{";
            for (unsigned y = 0; y < K; ++y)
              cout << (y ? ", " : "") << '"' << labels[y] << "\": " << predictions[i].posterior[y];
            cout << '}';
          }
          cout << endl;
        }
      }
    }

    if (!dont_write_weights) {
      cout << p << "\t***CATEGORICAL***";
      for (unsigned y = 0; y < K; ++y)
        cout << '\t' << labels[y];
      cout << endl;
      for (unsigned y = 0; y < km1; ++y)
        cout << labels[y] << "\t***BIAS***\t" << weights[y] << endl;
      for (unsigned y = 0; y < km1; ++y) {
        for (unsigned f = 0; f < p; ++f) {
          const double w = weights[km1 + y * p + f];
          if (w)
            cout << labels[y] << "\t" << FD::Convert(f) << "\t" << w << endl;
        }
      }
    }
  }

  return 0;
}
