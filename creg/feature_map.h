#ifndef _FEATURE_MAP_H_
#define _FEATURE_MAP_H_

#include <cstdlib>
#include <cstddef>
#include <utility>

class FeatureMapStorage;

// represents a feature map for some instance
// memory is owned by a FeatureMapStorage object
class FrozenFeatureMap {
 public:
  FrozenFeatureMap() {}
  FrozenFeatureMap(const FeatureMapStorage* fms, unsigned i) :
      fms_(fms), pos_(i) {}
  typedef const std::pair<int,float>* const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
 private:
  const FeatureMapStorage* fms_;
  unsigned pos_;
};

// fast dynamic storage for FeatureMaps
//  TODO store indices/offsets
//  TODO back with a file
class FeatureMapStorage {
 public:
  friend class FrozenFeatureMap;
  FeatureMapStorage() :
      dlen_(),
      olen_(),
      dcapacity_(1024),
      ocapacity_(64),
      data_(static_cast<std::pair<int,float>*>(malloc(dcapacity_ * sizeof(std::pair<int,float>)))),
      offsets_(static_cast<size_t*>(malloc(ocapacity_ * sizeof(size_t)))) {}
  ~FeatureMapStorage() { free(data_); }
  // copy the feature map pointed to by [begin,end) into the storage
  FrozenFeatureMap AddFeatureMap(const std::pair<int,float>* begin,
                                 const std::pair<int,float>* end) {
    const size_t msize = end - begin;
    size_t cur = dlen_;
    dlen_ += msize;
    if (dlen_ > dcapacity_)
      EnsureCapacity(dlen_);
    for (size_t i = 0; i < msize; ++i) {
      data_[cur + i] = *begin;
      ++begin;
    }
    PushOffset(cur + msize);
    return FrozenFeatureMap(this, olen_ - 1);
  }
 private:
  void PushOffset(size_t offset) {
    if (olen_ == ocapacity_) {
      const size_t ns = olen_ * 1.5;
      offsets_ = static_cast<size_t*>(realloc(offsets_, ns * sizeof(size_t)));
      ocapacity_ = ns;
    }
    offsets_[olen_] = offset;
    ++olen_;
  }
  void EnsureCapacity(size_t minsize) {
    const size_t ns = minsize * 1.5;
    data_ = static_cast<std::pair<int,float>*>(realloc(data_, ns * sizeof(std::pair<int, float>)));
    dcapacity_ = ns;
  }
  size_t dlen_;
  size_t olen_;
  size_t dcapacity_;
  size_t ocapacity_;
  std::pair<int,float>* data_;
  size_t* offsets_;
};

inline FrozenFeatureMap::const_iterator FrozenFeatureMap::begin() const {
  if (pos_ == 0) return &fms_->data_[0];
  return &fms_->data_[fms_->offsets_[pos_ - 1]];
}

inline FrozenFeatureMap::const_iterator FrozenFeatureMap::end() const {
  return &fms_->data_[fms_->offsets_[pos_]];
}

#endif
