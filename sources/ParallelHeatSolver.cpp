/**
 * @file    ParallelHeatSolver.cpp
 * 
 * @author  Ondrej Vlcek <xvlcek27@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties& simulationProps,
                                       const MaterialProperties&   materialProps)
: HeatSolverBase(simulationProps, materialProps)
{
  MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);

  initGridTopology();
  initDataDistribution();
  allocLocalTiles();
  initHaloExchange();
  

  if(!mSimulationProps.getOutputFileName().empty())
  {
    if (mSimulationProps.useParallelIO()) {
      openOutputFileParallel();
    }
    else {
      if (mWorldRank == 0) {
        openOutputFileSequential();
      }
    }
  }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
  deinitHaloExchange();
  deallocLocalTiles();
  deinitDataDistribution();
  deinitGridTopology();
}

std::string_view ParallelHeatSolver::getCodeType() const
{
  return codeType;
}

void ParallelHeatSolver::initGridTopology()
{
  cartDimensions = mSimulationProps.getDecomposition() == SimulationProperties::Decomposition::d2 ? 2 : 1;
  
  int grid[2];  //Y-X
  mSimulationProps.getDecompGrid(grid[1], grid[0]);

  // X-X
  if (cartDimensions == 1) {
    grid[0] = grid[1];
  }

  int periods[] = {false, false};
  MPI_Cart_create(MPI_COMM_WORLD, cartDimensions, grid, periods, false, &topologyComm);
  MPI_Comm_set_name(topologyComm, "Topology communicator");

  MPI_Cart_coords(topologyComm, mWorldRank, cartDimensions, cartCoords);
  //Swapping to X-Y, which is what I prefer
  if (cartDimensions == 2) {
    std::swap(cartCoords[0], cartCoords[1]);
  }

  if (shouldComputeMiddleColumnAverageTemperature()) {
    MPI_Comm_split(MPI_COMM_WORLD, 0, cartCoords[1], &middleColumnComm);
    MPI_Comm_set_name(middleColumnComm, "Middle column communicator");
  }
  else {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, cartCoords[1], &middleColumnComm);
  }

  // Neighbor calculation
  if (cartDimensions == 2) {
    MPI_Cart_shift(topologyComm, 1/*x*/, 1, neighborsX, neighborsX + 1);
    MPI_Cart_shift(topologyComm, 0/*y*/, 1, neighborsY, neighborsY + 1);
  }
  else {
    MPI_Cart_shift(topologyComm, 0/*x*/, 1, neighborsX, neighborsX + 1);
  }
}

void ParallelHeatSolver::deinitGridTopology()
{
  MPI_Comm_free(&topologyComm);
  if (middleColumnComm != MPI_COMM_NULL)
    MPI_Comm_free(&middleColumnComm);
}

void ParallelHeatSolver::initDataDistribution()
{
  int gridX = 0, gridY = 0;
  mSimulationProps.getDecompGrid(gridX, gridY);

  int domSize = mMaterialProps.getEdgeSize();

  tileSize[0] = domSize / gridX;
  tileSize[1] = domSize / gridY;

  innerTileSize[0] = tileSize[0] - 2 * haloZoneSize;
  innerTileSize[1] = tileSize[1] - 2 * haloZoneSize;

  outerTileSize[0] = tileSize[0] + 2 * haloZoneSize;
  outerTileSize[1] = tileSize[1] + 2 * haloZoneSize;

  innerOffset[0] = 2 * haloZoneSize;
  innerOffset[1] = 2 * haloZoneSize;

  MPI_Datatype globalIntType, globalFloatType;

  MPI_Type_vector(tileSize[1], tileSize[0], domSize, MPI_INT, &globalIntType);
  MPI_Type_vector(tileSize[1], tileSize[0], domSize, MPI_FLOAT, &globalFloatType);

  MPI_Type_create_resized(globalIntType, 0, sizeof(int), &resizedIntType);
  MPI_Type_commit(&resizedIntType);
  MPI_Type_create_resized(globalFloatType, 0, sizeof(float), &resizedFloatType);
  MPI_Type_commit(&resizedFloatType);

  MPI_Type_free(&globalFloatType);
  MPI_Type_free(&globalIntType);

  //matrix order, not XY order
  const int gsize[] = {outerTileSize[1], outerTileSize[0]};
  const int lsize[] = {tileSize[1], tileSize[0]};
  const int starts[] = {haloZoneSize,  haloZoneSize};
  
  MPI_Type_create_subarray(2, gsize, lsize, starts, MPI_ORDER_C, MPI_INT, &localIntType);
  MPI_Type_commit(&localIntType);
  MPI_Type_create_subarray(2, gsize, lsize, starts, MPI_ORDER_C, MPI_FLOAT, &localFloatType);
  MPI_Type_commit(&localFloatType);

}

void ParallelHeatSolver::deinitDataDistribution()
{
  MPI_Type_free(&resizedFloatType);
  MPI_Type_free(&resizedIntType);
  MPI_Type_free(&localFloatType);
  MPI_Type_free(&localIntType);
}

void ParallelHeatSolver::allocLocalTiles()
{
  temperature.resize(2 * localTileSize());
  materialProps.resize(localTileSize());
  materialType.resize(localTileSize());
}

void ParallelHeatSolver::deallocLocalTiles()
{
  //Should be deallocated by AlignedAllocator
}

void ParallelHeatSolver::initHaloExchange()
{
  MPI_Type_vector(tileSize[1], haloZoneSize, mStride(), MPI_FLOAT, &haloColumn);
  MPI_Type_commit(&haloColumn);
  MPI_Type_vector(haloZoneSize, tileSize[0], mStride(), MPI_FLOAT, &haloRow);
  MPI_Type_commit(&haloRow);

  if (mSimulationProps.isRunParallelRMA())
  {
    MPI_Win_create(temperature.data(),
                  static_cast<MPI_Aint>(sizeof(float) * localTileSize()),
                  sizeof(float), MPI_INFO_NULL, topologyComm, &windowPrev);
    MPI_Win_set_name(windowPrev, "Halo window A");

    MPI_Win_create(temperature.data() + localTileSize(),
                  static_cast<MPI_Aint>(sizeof(float) * localTileSize()),
                  sizeof(float), MPI_INFO_NULL, topologyComm, &windowNext);
    MPI_Win_set_name(windowNext, "Halo window B");
  }
}

void ParallelHeatSolver::deinitHaloExchange()
{
  MPI_Type_free(&haloColumn);
  MPI_Type_free(&haloRow);

  if (mSimulationProps.isRunParallelRMA()) {
    MPI_Win_free(&windowPrev);
    MPI_Win_free(&windowNext);
  }
}

template<typename T>
void ParallelHeatSolver::scatterTiles(const T* globalData, T* localData)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

  const MPI_Datatype localType = std::is_same_v<T, int> ? localIntType : localFloatType;
  const MPI_Datatype resizedType = std::is_same_v<T, int> ? resizedIntType : resizedFloatType;
  
  int gridX, gridY;
  mSimulationProps.getDecompGrid(gridX, gridY);

  int displs[gridX * gridY];
  int sends[gridX * gridY];

  for (int y = 0; y < gridY; y++) {
    for (int x = 0; x < gridX; x++) {
      displs[y * gridX + x] = y * gridX * tileSize[0] * tileSize[1] + x * tileSize[0];
      sends[y * gridX + x] = 1;
    }
  }

  MPI_Scatterv(globalData, sends, displs, resizedType, localData, 1, localType, 0, topologyComm);
}

template<typename T>
void ParallelHeatSolver::gatherTiles(const T* localData, T* globalData)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

  const MPI_Datatype localType = std::is_same_v<T, int> ? localIntType : localFloatType;
  const MPI_Datatype resizedType = std::is_same_v<T, int> ? resizedIntType : resizedFloatType;

  int gridX, gridY;
  mSimulationProps.getDecompGrid(gridX, gridY);

  int displs[gridX * gridY];
  int sends[gridX * gridY];

  for (int y = 0; y < gridY; y++)
  {
    for (int x = 0; x < gridX; x++)
    {
      displs[y * gridX + x] = y * gridX * tileSize[0] * tileSize[1] + x * tileSize[0];
      sends[y * gridX + x] = 1;
    }
  }

  MPI_Gatherv(localData, 1, localType, globalData, sends, displs, resizedType, 0, topologyComm);
}

void ParallelHeatSolver::computeHaloZones(const float* oldTemp, float* newTemp)
{
  int gridX, gridY;
  mSimulationProps.getDecompGrid(gridX, gridY);
  

  // edge check -> no halo zone calculation along borders
  bool hasLeft = cartCoords[0] != 0;
  bool hasRight = cartCoords[0] != gridX - 1;
  bool hasTop = cartCoords[1] != 0;
  bool hasBottom = cartCoords[1] != gridY - 1;

  if (cartDimensions == 2)
  {
    // top-down
    if (hasTop)
    {
      updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                 innerOffset[0], haloZoneSize,
                 innerTileSize[0], haloZoneSize, mStride());
      if (hasLeft) {
        updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                   haloZoneSize, haloZoneSize,
                   haloZoneSize, haloZoneSize, mStride());
      }
      if (hasRight) {
        updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                   innerOffset[0] + innerTileSize[0], haloZoneSize,
                   haloZoneSize, haloZoneSize, mStride());
      }
    }
    
    if (hasBottom)
    {
      updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                 innerOffset[0], innerOffset[1] + innerTileSize[1],
                 innerTileSize[0], haloZoneSize, mStride());
      if (hasLeft)
      {
        updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                   haloZoneSize, innerOffset[1] + innerTileSize[1],
                   haloZoneSize, haloZoneSize, mStride());
      }
      if (hasRight)
      {
        updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
                   innerOffset[0] + innerTileSize[0], innerOffset[1] + innerTileSize[1],
                   haloZoneSize, haloZoneSize, mStride());
      }
    }
  }

  //left-right
  if (hasLeft) {
    updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
              haloZoneSize, innerOffset[1],
              haloZoneSize, innerTileSize[1], mStride());
  }
  if (hasRight) {
    updateTile(oldTemp, newTemp, materialProps.data(), materialType.data(),
              innerOffset[0] + innerTileSize[0], innerOffset[1],
              haloZoneSize, innerTileSize[1], mStride());
  }
}

void ParallelHeatSolver::startHaloExchangeP2P(float* localData, std::array<MPI_Request, 8>& requests)
{
  // left
  MPI_Isend(localData + getOffset(haloZoneSize, haloZoneSize),
            1, haloColumn, neighborsX[0], 0, topologyComm, &requests[0]);
  // right
  MPI_Isend(localData + getOffset(outerTileSize[0] - 2 * haloZoneSize, haloZoneSize),
            1, haloColumn, neighborsX[1], 0, topologyComm, &requests[1]);

  // left
  MPI_Irecv(localData + getOffset(0, haloZoneSize),
            1, haloColumn, neighborsX[0], 0, topologyComm, &requests[2]);
  // right
  MPI_Irecv(localData + getOffset(outerTileSize[0] - haloZoneSize, haloZoneSize),
            1, haloColumn, neighborsX[1], 0, topologyComm, &requests[3]);

  if (cartDimensions == 2) {
    // top
    MPI_Isend(localData + getOffset(haloZoneSize, haloZoneSize),
              1, haloRow, neighborsY[0], 0, topologyComm, &requests[4]);
    // bottom
    MPI_Isend(localData + getOffset(haloZoneSize, outerTileSize[1] - 2 * haloZoneSize),
              1, haloRow, neighborsY[1], 0, topologyComm, &requests[5]);
    // top
    MPI_Irecv(localData + getOffset(haloZoneSize, 0),
              1, haloRow, neighborsY[0], 0, topologyComm, &requests[6]);
    // bottom
    MPI_Irecv(localData + getOffset(haloZoneSize, outerTileSize[1] - haloZoneSize),
              1, haloRow, neighborsY[1], 0, topologyComm, &requests[7]);
  }
  else {
    requests[4] = MPI_REQUEST_NULL;
    requests[5] = MPI_REQUEST_NULL;
    requests[6] = MPI_REQUEST_NULL;
    requests[7] = MPI_REQUEST_NULL;
  }
}

void ParallelHeatSolver::startHaloExchangeRMA(float* localData, MPI_Win window)
{
  MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT, window); //open

  // left
  MPI_Get(localData + getOffset(0, haloZoneSize),
          1, haloColumn, neighborsX[0],
          getOffset(outerTileSize[0] - 2 * haloZoneSize, haloZoneSize),
          1, haloColumn, window);

  // right
  MPI_Get(localData + getOffset(outerTileSize[0] - haloZoneSize, haloZoneSize),
          1, haloColumn, neighborsX[1],
          getOffset(haloZoneSize, haloZoneSize),
          1, haloColumn, window);

  if (cartDimensions == 2) {
    // top
    MPI_Get(localData + getOffset(haloZoneSize, 0),
            1, haloRow, neighborsY[0],
            getOffset(haloZoneSize, outerTileSize[1] - 2 * haloZoneSize),
            1, haloRow, window);

    // bottom
    MPI_Get(localData + getOffset(haloZoneSize, outerTileSize[1] - haloZoneSize),
            1, haloRow, neighborsY[1],
            getOffset(haloZoneSize, haloZoneSize),
            1, haloRow, window);
  }
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8>& requests)
{
  MPI_Waitall(8, std::begin(requests), MPI_STATUSES_IGNORE);
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window)
{
  MPI_Win_fence(MPI_MODE_NOPUT, window); //close
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>>& outResult)
{
  std::array<MPI_Request, 8> requestsP2P{};

  scatterTiles(&mMaterialProps.getInitialTemperature()[0], temperature.data());
  scatterTiles(&mMaterialProps.getDomainMap()[0], materialType.data());
  scatterTiles(&mMaterialProps.getDomainParameters()[0], materialProps.data());

  startHaloExchangeP2P(temperature.data(), requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);
  startHaloExchangeP2P(materialProps.data(), requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);

  std::copy(temperature.data(), temperature.data() + localTileSize(), temperature.data() + localTileSize());

  double startTime = MPI_Wtime();

  float* tempBuffer[] = {temperature.data(), temperature.data() + localTileSize()};
  MPI_Win windows[] = {windowPrev, windowNext};

  // 3. Start main iterative simulation loop.
  for(std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
  {
    const std::size_t oldIdx = iter % 2;       // Index of the buffer with old temperatures
    const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

  computeHaloZones(tempBuffer[oldIdx], tempBuffer[newIdx]);

  if (mSimulationProps.isRunParallelRMA()) {
    startHaloExchangeRMA(tempBuffer[newIdx], windows[newIdx]);
  }
  else {
    startHaloExchangeP2P(tempBuffer[newIdx], requestsP2P);
  }

  updateTile(tempBuffer[oldIdx], tempBuffer[newIdx], materialProps.data(), materialType.data(),
            innerOffset[0], innerOffset[1], innerTileSize[0], innerTileSize[1], mStride());

  if (mSimulationProps.isRunParallelRMA()) {
    awaitHaloExchangeRMA(windows[newIdx]);
  }
  else {
    awaitHaloExchangeP2P(requestsP2P);
  }

  if (shouldStoreData(iter))
  {
    if (mSimulationProps.useParallelIO())
    {
      storeDataIntoFileParallel(mFileHandle, iter, tempBuffer[newIdx]);
    }
    else
    {
      gatherTiles(tempBuffer[newIdx], &outResult[0]);

      if (mWorldRank == 0) {
        storeDataIntoFileSequential(mFileHandle, iter, outResult.data());
      }
    }

  }
    
  if(shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
  {
    float avg = computeMiddleColumnAverageTemperatureParallel(tempBuffer[newIdx]);
    int mccRank;
    MPI_Comm_rank(middleColumnComm, &mccRank);
    if (mccRank == 0) {
      printProgressReport(iter, avg);
    }
  }
  } //for

  const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures

  double elapsedTime = MPI_Wtime() - startTime;

  gatherTiles(tempBuffer[resIdx], &outResult[0]);

  if (mWorldRank == 0) {
    float avg = computeMiddleColumnAverageTemperatureSequential(&outResult[0]);
    printFinalReport(elapsedTime, avg);
  }
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
  int gridX, gridY;
  mSimulationProps.getDecompGrid(gridX, gridY);
  return cartCoords[0] == gridX / 2;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const
{
  float avg = 0.f;

  const int stride = mStride();
  const int offset = mMaterialProps.getEdgeSize() / 2 - cartCoords[0] * tileSize[0] + haloZoneSize * stride + haloZoneSize;
  const int count = tileSize[1];

  #pragma omp parallel for reduction(+ : avg)
  for (int i = 0; i < count; i++)
  {
    avg += localData[i * stride + offset];
  }

  float gavg = 0;

  MPI_Reduce(&avg, &gavg, 1, MPI_FLOAT, MPI_SUM, 0, middleColumnComm);

  return gavg / mMaterialProps.getEdgeSize();
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const
{

  float avg = 0.f;
  const int edge = mMaterialProps.getEdgeSize();
  #pragma omp parallel for reduction(+:avg)
  for (int i = 0; i < edge; i++) {
    avg += globalData[i * edge + edge / 2];
  }

  return avg / edge;
}

void ParallelHeatSolver::openOutputFileSequential()
{
  // Create the output file for sequential access.
  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(!mFileHandle.valid())
  {
    throw std::ios::failure("Cannot create output file!");
  }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t        fileHandle,
                                                     std::size_t  iteration,
                                                     const float* globalData)
{
  storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel()
{
#ifdef H5_HAVE_PARALLEL
  Hdf5PropertyListHandle faplHandle{};
  
  faplHandle = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(faplHandle, topologyComm, MPI_INFO_NULL);
  H5Pset_alignment(faplHandle, 524288, 262144);

  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          faplHandle);
  if(!mFileHandle.valid())
  {
    throw std::ios::failure("Cannot create output file!");
  }
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t                         fileHandle,
                                                   [[maybe_unused]] std::size_t  iteration,
                                                   [[maybe_unused]] const float* localData)
{
  if (fileHandle == H5I_INVALID_HID)
  {
    return;
  }

#ifdef H5_HAVE_PARALLEL
  std::array gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
                      static_cast<hsize_t>(mMaterialProps.getEdgeSize())};

  // Create new HDF5 group in the output file
  std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());

  Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  {

    hsize_t tileSize[] = {
      static_cast<hsize_t>(tileSize[1]),
      static_cast<hsize_t>(tileSize[0])
    };

    hsize_t localTileSize[] = {
      static_cast<hsize_t>(outerTileSize[1]),
      static_cast<hsize_t>(outerTileSize[0])
    };

    hsize_t gstarts[] = { cartCoords[1] * tileSize[0], cartCoords[0] * tileSize[1] };

    hsize_t lstarts[] = { haloZoneSize, haloZoneSize };

    // Create new dataspace and dataset using it.
    static constexpr std::string_view dataSetName{"Temperature"};

    Hdf5PropertyListHandle datasetPropListHandle{};

    datasetPropListHandle = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(datasetPropListHandle, 2, tileSize);

    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, datasetPropListHandle,
                                              H5P_DEFAULT)); 

    Hdf5DataspaceHandle memSpaceHandle{};

  memSpaceHandle = H5Screate_simple(2, localTileSize, nullptr);

  int decompX, decompY;
  mSimulationProps.getDecompGrid(decompX, decompY);

  hsize_t strides[] = { 1, 1 };

  H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, gstarts, strides, tileSize, NULL);

  H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, lstarts, strides, tileSize, NULL);

    Hdf5PropertyListHandle propListHandle{};
    
    propListHandle = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
  }

  {
    // 3. Store attribute with current iteration number in the group.
    static constexpr std::string_view attributeName{"Time"};
    Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
    Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                   H5T_IEEE_F64LE, dataSpaceHandle,
                                                   H5P_DEFAULT, H5P_DEFAULT));
    const double snapshotTime = static_cast<double>(iteration);
    H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
  }
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
