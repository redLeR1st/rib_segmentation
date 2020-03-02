/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/

#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <functional>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"

#include "itkBinaryThresholdImageFilter.h"

// Software Guide : BeginCodeSnippet
#include "itkHoughTransform2DCirclesImageFilter.h"
// Software Guide : EndCodeSnippet
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkThresholdImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include <list>
#include "itkCastImageFilter.h"
#include "itkMath.h"

#include "itkFlipImageFilter.h"

#include "itkTileImageFilter.h"

#include <itkSpatialObject.h>

#include <itkCropImageFilter.h>

#include "itkScaleTransform.h"

#include "itkResampleImageFilter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"
#include "itkImageDuplicator.h"

#include "itkOtsuMultipleThresholdsImageFilter.h"

#include "itkEllipseSpatialObject.h"

#include "itkRegionGrowImageFilter.h"
#include "itkNeighborhoodConnectedImageFilter.h"

#include "itkScalarImageToHistogramGenerator.h"

#include <itkOtsuMultipleThresholdsCalculator.h>

#include "vector"

#include "itkAddImageFilter.h"

#include "itkImageDuplicator.h"

#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkMedianImageFilter.h"

#include "itkConnectedComponentImageFilter.h"

#include "itkLabelShapeKeepNObjectsImageFilter.h"

#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkLabelImageToStatisticsLabelMapFilter.h"

#include "itkLabelMapToLabelImageFilter.h"

#include "itkObjectByObjectLabelMapFilter.h"

#include "itkLabelUniqueLabelMapFilter.h"

using PixelType = short;

using ImageType = itk::Image< PixelType, 3 >;
using ImageType2D = itk::Image< PixelType, 2 >;

using ReaderType = itk::ImageFileReader< ImageType >;
using WriterType = itk::ImageFileWriter< ImageType >;

typedef itk::SpatialObject<2>             SpatialObjectType;
typedef SpatialObjectType::TransformType  TransformType;
typedef   float           AccumulatorPixelType;

typedef itk::EllipseSpatialObject<2> ElipseType;

typedef itk::HoughTransform2DCirclesImageFilter<PixelType,
    AccumulatorPixelType> HoughTransformFilterType;
typedef HoughTransformFilterType::CirclesListType CirclesListType;
//typedef HoughTransformFilterType::CircleType::Type CirclesListType;

typedef itk::OtsuMultipleThresholdsImageFilter <ImageType2D, ImageType2D> OtsuFilterType2D;
typedef itk::OtsuMultipleThresholdsImageFilter <ImageType, ImageType> OtsuFilterType;

typedef itk::RegionGrowImageFilter<ImageType, ImageType> RegionGrowImageFilterType;
typedef itk::NeighborhoodConnectedImageFilter<ImageType, ImageType > ConnectedFilterType;

using ScalarImageToHistogramGeneratorType = itk::Statistics::ScalarImageToHistogramGenerator<ImageType>;
using HistogramType = ScalarImageToHistogramGeneratorType::HistogramType;

using CalculatorType = itk::OtsuMultipleThresholdsCalculator<HistogramType>;

using ConnectedComponentImageFilterType = itk::ConnectedComponentImageFilter<ImageType, ImageType>;

CalculatorType::OutputType otusHist(ScalarImageToHistogramGeneratorType::Pointer input);

OtsuFilterType::ThresholdVectorType otusImg(ReaderType::Pointer input);

ImageType::IndexType get_optimal_seed(ImageType::Pointer ribs_only, std::vector<ImageType::IndexType> seed_vec, int begin, int end);

ScalarImageToHistogramGeneratorType::Pointer normalize(ImageType::Pointer input);

using AddImageFilterType = itk::AddImageFilter<ImageType, ImageType>;

using DuplicatorType = itk::ImageDuplicator<ImageType>;

using LabelShapeKeepNObjectsImageFilterType = itk::LabelShapeKeepNObjectsImageFilter<ImageType>;


using ShapeLabelObjectType = itk::ShapeLabelObject<unsigned short, 3>;
using LabelMapType = itk::LabelMap<ShapeLabelObjectType>;

using StatisticsShapeLabelObjectType = itk::StatisticsLabelObject<unsigned short, 3>;
using StatisticsLabelMapType = itk::LabelMap<StatisticsShapeLabelObjectType>;

using I2LType = itk::LabelImageToShapeLabelMapFilter<ImageType, LabelMapType>;
using I2LType_stat = itk::LabelImageToStatisticsLabelMapFilter<ImageType, StatisticsLabelMapType>;

typedef itk::LabelImageToLabelMapFilter< ImageType, LabelMapType > LabelImageToLabelMapFilterType;

using LabelMapToLabelImageFilterType = itk::LabelMapToLabelImageFilter<LabelMapType, ImageType>;

bool circle_itersect(double x1, double y1, double r1, double x2, double y2, double r2) {
    double distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    double radSumSq_m = (r1 - r2) * (r1 - r2);
    double radSumSq_p = (r1 + r2) * (r1 + r2);

    if (radSumSq_m <= distSq && distSq <= radSumSq_p) { // intersect
        return true;
    }
    else     // not intersect
        return false;
}

/**
 * Corricates the circle list globaly
 *
 * @param circles_list list of the circles 
 */
void corrigate_circles(std::vector<CirclesListType> &circles_list) {

    TransformType::OffsetType Object2ToObject1Offset;
    Object2ToObject1Offset[0] = 100;
    Object2ToObject1Offset[1] = 100;

    double av_x = 0;
    double av_y = 0;
    double av_rad = 0;

    int sum = 0;

    auto it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();
        while (itCircles != (*it).end())
        {


            av_x += (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
            av_y += (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
            av_rad += (*itCircles)->GetRadius()[0];

            sum++;
            itCircles++;
        }
        it++;
    }
    av_x /= sum;
    av_y /= sum;
    av_rad /= sum;

    std::cerr << "AVG_X " << av_x << std::endl;
    std::cerr << "AVG_Y " << av_y << std::endl;
    std::cerr << "AVG_RAD " << av_rad << std::endl;

    int filter_size = 40;

    it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();

        //std::cout << "x1 " << (*itCircles)->GetObjectToParentTransform()->GetOffset()[0] << std::endl;
        //std::cout << "y1 " << (*itCircles)->GetObjectToParentTransform()->GetOffset()[1] << std::endl;
        //std::cout << "r1 " << (*itCircles)->GetRadius()[0]<< std::endl;
        //
        //std::cout << "x2 " << (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0] << std::endl;
        //std::cout << "y2 " << (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1] << std::endl;
        //std::cout << "r2 " << (*(*it2).begin())->GetRadius()[0] << std::endl;
        //std::cout << "INTER " << circle_itersect(av_x, av_y, av_rad,26,172,10) << std::endl;
        //std::cout << "NOT INTER " << circle_itersect(3,4,0,14,18,0) << std::endl;


        //     if (!circle_itersect(av_x, av_y, av_rad, (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0],
        //         (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1], (*(*it2).begin())->GetRadius()[0])) {
        if (!circle_itersect((*itCircles)->GetObjectToParentTransform()->GetOffset()[0], (*itCircles)->GetObjectToParentTransform()->GetOffset()[1], (*itCircles)->GetRadius()[0],
            av_x, av_y, av_rad)) {

            // std::cout << "CORRIGATE NEEDED " << std::endl;

            int min_dist = 99999999;
            while (++itCircles != (*it).end()) {
                int x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
                int y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
                double dist = (x - av_x) * (x - av_x) + (y - av_y) * (y - av_y);

                if (min_dist > dist) {

                    CirclesListType::const_iterator first = (*it).begin();

                    Object2ToObject1Offset[0] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
                    Object2ToObject1Offset[1] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
                    (*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
                    (*first)->SetRadius((*itCircles)->GetRadius()[0]);
                    min_dist = dist;
                }

                //if (circle_itersect((*itCircles)->GetObjectToParentTransform()->GetOffset()[0], (*itCircles)->GetObjectToParentTransform()->GetOffset()[1], (*itCircles)->GetRadius()[0],
                //    av_x, av_y, av_rad)) {
                //    
                //    CirclesListType::const_iterator first = (*it).begin();
                //
                //    Object2ToObject1Offset[0] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
                //    Object2ToObject1Offset[1] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
                //    (*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
                //    (*first)->SetRadius((*itCircles)->GetRadius()[0]);
                //    break;
                //
                //}
            }

            // MEDIAN METHOD!
            // ///////////////////////////////////////////
            // std::map<int, double> dis_from_avg;
            // int key = 0;
            // 
            // bool flag = false;
            // auto it2 = std::next(it, 0);
            // //for (int i = 0; i < filter_size; i++) {
            // while (key < filter_size) {
            //     if (++it2 == circles_list.end()) {
            //         flag = true;
            //         break;
            //     }
            //     else {
            //         int x = (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0];
            //         int y = (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1];
            //         int rad = (*(*it2).begin())->GetRadius()[0];
            // 
            //         if (circle_itersect(x, y, rad, av_x, av_y, av_rad)) {
            //             double dist = (x - av_x) * (x - av_x) + (y - av_y) * (y - av_y);
            //             dis_from_avg.insert(std::make_pair(key, dist));
            //             key++;
            //         }
            // 
            //     }
            // }
            // if (flag) {
            //     break;
            // }
            // typedef std::pair<int, double> pair;
            // // create a empty vector of pairs
            // std::vector<pair> vec;
            // 
            // // copy key-value pairs from the map to the vector
            // std::copy(dis_from_avg.begin(),
            //     dis_from_avg.end(),
            //     std::back_inserter<std::vector<pair>>(vec));
            // 
            // // sort the vector by increasing order of its pair's second value
            // // if second value are equal, order by the pair's first value
            // std::sort(vec.begin(), vec.end(),
            //     [](const pair& l, const pair& r) {
            //     if (l.second != r.second)
            //         return l.second < r.second;
            // 
            //     return l.first < r.first;
            // });
            // 
            // // print the vector
            // for (auto const &pair : vec) {
            //     std::cout << '{' << pair.first << "," << pair.second << '}' << '\n';
            // }
            // 
            // auto median_key_iterator = std::next(vec.begin(), filter_size / 3);
            // 
            // int median_circle_key = (*median_key_iterator).first;
            // 
            // auto median_circle = std::next(it, median_circle_key);
            // 
            // CirclesListType::const_iterator itCircles = (*it).begin();
            // 
            // Object2ToObject1Offset[0] = (*(*median_circle).begin())->GetObjectToParentTransform()->GetOffset()[0];
            // Object2ToObject1Offset[1] = (*(*median_circle).begin())->GetObjectToParentTransform()->GetOffset()[1];
            // (*itCircles)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
            // (*itCircles)->SetRadius((*(*median_circle).begin())->GetRadius()[0]);
            // 
            // 
            // ///////////////////////////////////////////
            // 
            // 
            // 
            // //do {
            // //
            // //    Object2ToObject1Offset[0] = (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0];
            // //    Object2ToObject1Offset[1] = (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1];
            // //    (*itCircles)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
            // //    (*itCircles)->SetRadius((*(*it2).begin())->GetRadius()[0]);
            // //
            // //    if (++it2 == circles_list.end()) {
            // //        break;
            // //    }
            // //}// while (circle_itersect(av_x, av_y, av_rad, (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0],
            // // //   (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1], (*(*it2).begin())->GetRadius()[0]));
            // //while (!circle_itersect((*itCircles)->GetObjectToParentTransform()->GetOffset()[0], (*itCircles)->GetObjectToParentTransform()->GetOffset()[1], (*itCircles)->GetRadius()[0], (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[0],
            // //    (*(*it2).begin())->GetObjectToParentTransform()->GetOffset()[1], (*(*it2).begin())->GetRadius()[0]));

        }
        //CirclesListType::const_iterator h = (*it).begin();
        // lets see where is the average
        //Object2ToObject1Offset[0] = av_x;
        //Object2ToObject1Offset[1] = av_y;
        //(*h)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
        //(*h)->SetRadius(av_rad);
        it++;

    }

}

/**
 * Draws the best fitting circle on each axial slice of the image
 *
 * @param input the input image
 * @param circles_list the list of the circles
 */
void draw_circle(ImageType::Pointer input, std::vector<CirclesListType> circles_list) {

    ImageType::IndexType localIndex3D;

    const ImageType * inputImage = input;
    ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    int tolerance = 100;
    int actual_slice_number = 0;

    std::cerr << "LIST SIZE " << circles_list.size() << std::endl;
    auto it = circles_list.begin();
    while (it != circles_list.end())
    {

        CirclesListType::const_iterator itCircles = (*it).begin();
        //while (itCircles != (*it).end())
        //{

        for (unsigned short x = 0; x < size[0]; x++) {
            for (unsigned short y = 0; y < size[1]; y++) {
                int x_cube = (x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0])*(x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0]);
                int y_cube = (y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1])*(y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1]);

                // keep only inside
                if ((x_cube + y_cube) >((*itCircles)->GetRadius()[0] + tolerance) * ((*itCircles)->GetRadius()[0]) + tolerance) {
                    localIndex3D[0] = x;
                    localIndex3D[1] = y;
                    localIndex3D[2] = actual_slice_number;
                    input->SetPixel(localIndex3D, 0);
                    //input->GetOutput()->SetPixel(localIndex3D, -1);
                }

                // on the circle, then we draw
                /*  tolerance = 0;
                int circle_line_tik = 65;
                if ((x_cube + y_cube) > ((*itCircles)->GetRadius()[0] + tolerance) * ((*itCircles)->GetRadius()[0]) + tolerance - circle_line_tik && (x_cube + y_cube) < ((*itCircles)->GetRadius()[0] + tolerance) * ((*itCircles)->GetRadius()[0]) + tolerance + circle_line_tik) {
                localIndex3D[0] = x;
                localIndex3D[1] = y;
                localIndex3D[2] = actual_slice_number;
                input->GetOutput()->SetPixel(localIndex3D, -10000);
                }*/

            }
        }
        //itCircles++;
        //}
        actual_slice_number++;
        it++;
    }
}

/**
 * Calculates the circles on the input image 
 *
 * @param input 2D image to search the circles for
 * @param original 3d image
 * @param keep_only_inside if true the values outside of the circles will be set to 0
 * @param circles_list conatining the result for each axial slices of the image
 *
 * @return circles_list conatining the result for each axial slices of the image
 */
ImageType2D::Pointer do_hough_on_image(ImageType2D::Pointer input, ImageType::Pointer original, bool keep_only_inside, std::vector<CirclesListType> &circles_list) {
    //#########################################

#if 0
    if (argc < 6)
    {
        std::cerr << "Missing Parameters " << std::endl;
        std::cerr << "Usage: " << argv[0] << std::endl;
        std::cerr << " inputImage " << std::endl;      1
            std::cerr << " outputImage" << std::endl;      2
            std::cerr << " numberOfCircles " << std::endl; 3
            std::cerr << " radius Min " << std::endl;      4
            std::cerr << " radius Max " << std::endl;      5
            std::cerr << " sweep Angle (default = 0)" << std::endl; 6
            std::cerr << " SigmaGradient (default = 1) " << std::endl; 7
            std::cerr << " variance of the accumulator blurring (default = 5) " << std::endl; 8
            std::cerr << " radius of the disk to remove from the accumulator (default = 10) " << std::endl; 9
            return EXIT_FAILURE;
    }

#endif


    ImageType2D::IndexType localIndex;
    //ImageType::IndexType localIndex3D;
    typedef itk::Image< AccumulatorPixelType, 2 > AccumulatorImageType;

    ImageType2D::Pointer localImage = input;

    //  Software Guide : BeginLatex
    //
    //  We create the HoughTransform2DCirclesImageFilter based on the pixel
    //  type of the input image (the resulting image from the
    //  ThresholdImageFilter).
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet
    // TODO: std::cout << "Computing Hough Map on slice: " << actual_slice_number << std::endl;

    HoughTransformFilterType::Pointer houghFilter
        = HoughTransformFilterType::New();
    // Software Guide : EndCodeSnippet
    //  Software Guide : BeginLatex
    //
    //  We set the input of the filter to be the output of the
    //  ImageFileReader. We set also the number of circles we are looking for.
    //  Basically, the filter computes the Hough map, blurs it using a certain
    //  variance and finds maxima in the Hough map. After a maximum is found,
    //  the local neighborhood, a circle, is removed from the Hough map.
    //  SetDiscRadiusRatio() defines the radius of this disc proportional to
    //  the radius of the disc found.  The Hough map is computed by looking at
    //  the points above a certain threshold in the input image. Then, for each
    //  point, a Gaussian derivative function is computed to find the direction
    //  of the normal at that point. The standard deviation of the derivative
    //  function can be adjusted by SetSigmaGradient(). The accumulator is
    //  filled by drawing a line along the normal and the length of this line
    //  is defined by the minimum radius (SetMinimumRadius()) and the maximum
    //  radius (SetMaximumRadius()).  Moreover, a sweep angle can be defined by
    //  SetSweepAngle() (default 0.0) to increase the accuracy of detection.
    //
    //  The output of the filter is the accumulator.
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet

    houghFilter->SetInput(input);
    //houghFilter->SetNumberOfCircles(atoi(argv[3]));
    houghFilter->SetNumberOfCircles(10);

    houghFilter->SetMinimumRadius(10);
    houghFilter->SetMaximumRadius(30);
    houghFilter->SetSweepAngle(1);
    houghFilter->SetSigmaGradient(3);
    houghFilter->SetVariance(2);

    //houghFilter->SetMinimumRadius(18);
    ////houghFilter->SetMinimumRadius(10);
    //houghFilter->SetMaximumRadius(25);
    //houghFilter->SetSweepAngle(0);
    //houghFilter->SetSigmaGradient(1.3);
    //houghFilter->SetVariance(2);
    //
    //houghFilter->SetDiscRadiusRatio(0);


    houghFilter->Update();
    AccumulatorImageType::Pointer localAccumulator = houghFilter->GetOutput();
    // Software Guide : EndCodeSnippet
    //  Software Guide : BeginLatex
    //
    //  We can also get the circles as \doxygen{EllipseSpatialObject}. The
    //  \code{GetCircles()} function return a list of those.
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet
    HoughTransformFilterType::CirclesListType circles;
    circles = houghFilter->GetCircles();
    // TODO: std::cout << "Found " << circles.size() << " circle(s)." << std::endl;
    // Software Guide : EndCodeSnippet
    //  Software Guide : BeginLatex
    //
    //  We can then allocate an image to draw the resulting circles as binary
    //  objects.
    //
    //  Software Guide : EndLatex
    // Software Guide : BeginCodeSnippet
    typedef  unsigned char                            OutputPixelType;
    typedef  itk::Image< PixelType, 2 > OutputImageType;
    ImageType2D::Pointer  localOutputImage = ImageType2D::New();
    ImageType2D::RegionType region;
    region.SetSize(input->GetLargestPossibleRegion().GetSize());
    region.SetIndex(input->GetLargestPossibleRegion().GetIndex());
    localOutputImage->SetRegions(region);
    localOutputImage->SetOrigin(input->GetOrigin());
    localOutputImage->SetSpacing(input->GetSpacing());
    localOutputImage->Allocate(true); // initializes buffer to zero
                                      // Software Guide : EndCodeSnippet
                                      //  Software Guide : BeginLatex
                                      //
                                      //  We iterate through the list of circles and we draw them.
                                      //
                                      //  Software Guide : EndLatex
                                      // Software Guide : BeginCodeSnippet

    CirclesListType::const_iterator itCircles = circles.begin();
    while (itCircles != circles.end())
    {
        // TODO: std::cout << "Center: ";
        // TODO: std::cout << (*itCircles)->GetObjectToParentTransform()->GetOffset()
        // TODO:     << std::endl;
        // TODO: std::cout << "Radius: " << (*itCircles)->GetRadius()[0] << std::endl;

        if (!keep_only_inside)
        {
            // Software Guide : EndCodeSnippet
            //  Software Guide : BeginLatex
            //
            //  We draw white pixels in the output image to represent each circle.
            //
            //  Software Guide : EndLatex
            // Software Guide : BeginCodeSnippet
            for (double angle = 0;
                angle <= itk::Math::twopi;
                angle += itk::Math::pi / 60.0)
            {
                typedef HoughTransformFilterType::CircleType::TransformType
                    TransformType;
                typedef TransformType::OutputVectorType
                    OffsetType;
                const OffsetType offset =
                    (*itCircles)->GetObjectToParentTransform()->GetOffset();
                localIndex[0] =
                    itk::Math::Round<long int>(offset[0]
                        + (*itCircles)->GetRadius()[0] * std::cos(angle));
                localIndex[1] =
                    itk::Math::Round<long int>(offset[1]
                        + (*itCircles)->GetRadius()[0] * std::sin(angle));
                OutputImageType::RegionType outputRegion =
                    localOutputImage->GetLargestPossibleRegion();
                if (outputRegion.IsInside(localIndex))
                {
                    input->SetPixel(localIndex, -2);
                }
            }
        }
        else {
            //const ImageType2D * inputImage = input->GetOutput();
            //ImageType2D::RegionType inputRegion = inputImage->GetBufferedRegion();
            //ImageType2D::SizeType size = inputRegion.GetSize();
            //int tolerance = 100;
            //
            //for (unsigned short x = 0; x < size[0]; x++) {
            //    for (unsigned short y = 0; y < size[1]; y++) {
            //        int x_cube = (x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0])*(x - (*itCircles)->GetObjectToParentTransform()->GetOffset()[0]);
            //        int y_cube = (y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1])*(y - (*itCircles)->GetObjectToParentTransform()->GetOffset()[1]);
            //        if ((x_cube + y_cube) > ((*itCircles)->GetRadius()[0] + tolerance ) * ((*itCircles)->GetRadius()[0]) + tolerance) {
            //            localIndex[0] = x;
            //            localIndex[1] = y;
            //            localIndex3D[0] = x;
            //            localIndex3D[1] = y;
            //            localIndex3D[2] = actual_slice_number;
            //            input->GetOutput()->SetPixel(localIndex, 0);
            //            original->GetOutput()->SetPixel(localIndex3D, 0);
            //        }
            //    }
            //}



        }
        itCircles++;
    }

    circles_list.push_back(circles);
    return input;

}

/**
* Corricates the circle list localy based on the average
*
* @param circles_list list of the circles
*/
void corrigate_circles_local(std::vector<CirclesListType> &circles_list1) {

    std::vector<CirclesListType> circles_list;

    for (int i = 0; i < circles_list1.size(); i++) {

        TransformType::OffsetType Object2ToObject1Offset;
        CirclesListType push_it;
        ElipseType::Pointer elipse = ElipseType::New();

        CirclesListType::const_iterator itCircles = circles_list1[i].begin();
        Object2ToObject1Offset[0] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
        Object2ToObject1Offset[1] = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
        elipse->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
        elipse->SetRadius((*itCircles)->GetRadius());
        push_it.push_back(elipse);;
        circles_list.push_back(push_it);
    }


    TransformType::OffsetType Object2ToObject1Offset;
    Object2ToObject1Offset[0] = 100;
    Object2ToObject1Offset[1] = 100;


    int filter_size = 15;

    for (int i = 0; i < circles_list.size(); i++) {


        double av_x = 0;
        double av_y = 0;
        double av_rad = 0;

        int sum = 0;
        int j = 0;
        //if (i < std::ceil(filter_size / 2)) {
        //    j = std::ceil(filter_size / 2);
        //}
        //else if (i >= circles_list.size() - std::ceil(filter_size / 2)) {
        //    j = circles_list.size() - std::ceil(filter_size / 2) - 1;
        //}
        //else {
        //    j = i;
        //}


        j = i - std::ceil(filter_size / 2);
        int end_of_filter_in_lopp = j + filter_size;
        for (; j < end_of_filter_in_lopp; j++) {
            if (j < 0 || j >= circles_list.size()) {
                continue;
            }
            CirclesListType::const_iterator itCircles = circles_list[j].begin();

            av_x += (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
            av_y += (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
            av_rad += (*itCircles)->GetRadius()[0];

            sum++;
        }
        av_x /= sum;
        av_y /= sum;
        av_rad /= sum;

        // AVG
        CirclesListType::const_iterator itCircles = circles_list[i].begin();
        double d = (((*itCircles)->GetObjectToParentTransform()->GetOffset()[0]) - (av_x))*(((*itCircles)->GetObjectToParentTransform()->GetOffset()[0]) - (av_x)) + (((*itCircles)->GetObjectToParentTransform()->GetOffset()[1]) - (av_y))*(((*itCircles)->GetObjectToParentTransform()->GetOffset()[1]) - (av_y));
        if (d < 100000) {


            CirclesListType::const_iterator first = circles_list1[i].begin();

            Object2ToObject1Offset[0] = av_x;
            Object2ToObject1Offset[1] = av_y;
            (*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
            (*first)->SetRadius(av_rad);


        }
        // AVG END

        // MED
        //j = 0;
        //if (i < std::ceil(filter_size / 2)) {
        //    j = std::ceil(filter_size / 2);
        //}
        //else if (i >= circles_list.size() - std::ceil(filter_size / 2)) {
        //    j = circles_list.size() - std::ceil(filter_size / 2) - 1;
        //}
        //else {
        //    j = i;
        //}
        //std::map<int, int> dist_map;
        //j = j - std::ceil(filter_size / 2);
        //end_of_filter_in_lopp = j + filter_size;
        //for (; j < end_of_filter_in_lopp; j++) {
        //    CirclesListType::const_iterator itCircles = circles_list[j].begin();
        //
        //    int x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
        //    int y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
        //    double dist = (x - av_x) * (x - av_x) + (y - av_y) * (y - av_y);
        //
        //    dist_map[j] = dist;
        //
        //}
        //
        //
        //// Declaring the type of Predicate that accepts 2 pairs and return a bool
        //typedef std::function<bool(std::pair<int, int>, std::pair<int, int>)> Comparator;
        //
        //// Defining a lambda function to compare two pairs. It will compare two pairs using second field
        //Comparator compFunctor =
        //    [](std::pair<int, int> elem1, std::pair<int, int> elem2)
        //{
        //    return elem1.second < elem2.second;
        //};
        //
        //// Declaring a set that will store the pairs using above comparision logic
        //std::set<std::pair<int, int>, Comparator> ordered_set(
        //    dist_map.begin(), dist_map.end(), compFunctor);
        //
        //// Iterate over a set using range base for loop
        //// It will display the items in sorted order of values
        //int med_x;
        //int med_y;
        //int med_rad;
        //int k = 0;
        //
        //for (std::pair<int, int> temp : ordered_set) {
        //   if (k == ordered_set.size() / 2) {
        //       CirclesListType::const_iterator itCircles = circles_list[k].begin();
        //
        //       med_x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];
        //       med_y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
        //       med_rad = (*itCircles)->GetRadius()[0];
        //       break;
        //   }
        //   k++;
        //}
        //
        //CirclesListType::const_iterator first = circles_list1[i].begin();
        //Object2ToObject1Offset[0] = med_x;
        //Object2ToObject1Offset[1] = med_y;
        //(*first)->GetObjectToParentTransform()->SetOffset(Object2ToObject1Offset);
        //(*first)->SetRadius(med_rad);

        // MED END

    }
}


ImageType::Pointer close_on_3d_slices(ImageType::Pointer input, int radius, int foreground) {
    std::cout << "Closing" << std::endl;
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();

    using BinaryMorphologicalClosingImageFilterType =
        itk::BinaryMorphologicalClosingImageFilter<ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
    closingFilter->SetForegroundValue(foreground);
    closingFilter->SetKernel(structuringElement);

    closingFilter->SetInput(input);
    closingFilter->Update();

    return closingFilter->GetOutput();
}

ImageType::Pointer close_on_2d_slices(ImageType::Pointer input, int radius, int foreground) {
    std::cout << "Closing" << std::endl;
    using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();
    
    using BinaryMorphologicalClosingImageFilterType =
        itk::BinaryMorphologicalClosingImageFilter<ImageType, ImageType, StructuringElementType>;
    BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
    closingFilter->SetForegroundValue(foreground);
    closingFilter->SetKernel(structuringElement);
    
    using PasteFilterType = itk::PasteImageFilter<ImageType, ImageType>;
    PasteFilterType::Pointer pasteFilter = PasteFilterType::New();
    
    const ImageType * inputImage = input;
    
    ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    int height_of_the_image;
    height_of_the_image = size[2];
    using ExtractFilterType = itk::ExtractImageFilter< ImageType, ImageType >;

    pasteFilter->SetDestinationImage(inputImage);
    for (int i = 1; i < height_of_the_image; i++) {
        
        ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
        extractFilter->SetDirectionCollapseToSubmatrix();
    
        size[2] = 1; // we extract along z direction
    
        ImageType::IndexType start = inputRegion.GetIndex();
    
        start[2] = i;
        ImageType::RegionType desiredRegion;
        desiredRegion.SetSize(size);
        desiredRegion.SetIndex(start);
        extractFilter->SetExtractionRegion(desiredRegion);;
        extractFilter->SetInput(inputImage);
    
        closingFilter->SetInput(extractFilter->GetOutput());
        pasteFilter->SetSourceImage(closingFilter->GetOutput());
        
        ImageType::SizeType indexRadius;
        indexRadius[0] = 1; // radius along x
        indexRadius[1] = 1; // radius along y
        indexRadius[2] = 0; // radius along z
        closingFilter->SetRadius(indexRadius);
        closingFilter->UpdateLargestPossibleRegion();
        closingFilter->Update();
        const ImageType * closingImage = closingFilter->GetOutput();
        pasteFilter->SetSourceRegion(closingImage->GetBufferedRegion());
        pasteFilter->SetDestinationIndex(start);
        
        pasteFilter->Update();
        pasteFilter->SetDestinationImage(pasteFilter->GetOutput());
    }
    std::cout << "Closing end" << std::endl;
    return pasteFilter->GetOutput();
}

// Vector of the seeds grown for left and right ribs
std::vector<ImageType::IndexType> seed_vec_global_l;
std::vector<ImageType::IndexType> seed_vec_global_r;

/**
 * Findig seed points and growing ribs or other objects from it
 *
 * @param reader_ribs image to search ribs for if @param is_cleaned_from_non_ribs is true this image should be the segmented and cleand image containing only ribs
 * @param number_of_test_file key number of the image
 * @param circles_list list of the circles
 * @param is_cleaned_from_non_ribs there if true we search seed points for labeling else we search seed points for region growing
 *
 * @return segmented image contains only the ribs
 */
ImageType::Pointer find_ribs(ImageType::Pointer reader_ribs, int number_of_test_file, std::vector<CirclesListType> circles_list, bool is_cleaned_from_non_ribs) {
    seed_vec_global_l.clear();
    seed_vec_global_r.clear();

    ImageType::Pointer image_to_process;

    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();

    typedef itk::CastImageFilter< ImageType, ImageType >
        CastingFilterType;
    CastingFilterType::Pointer caster = CastingFilterType::New();

    using FilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
    FilterType::Pointer threshFilter = FilterType::New();
    threshFilter = FilterType::New();
    threshFilter->SetInput(reader_ribs);
    
    if (!is_cleaned_from_non_ribs) {
        // RIB SEARCH
        std::cout << "Thresholding for ribs" << std::endl;


        OtsuFilterType::ThresholdVectorType thresholds;
        thresholds = otusHist(normalize(reader_ribs));
        for (unsigned int i = 0; i < thresholds.size(); i++)
        {
            std::cout << thresholds[i] << std::endl;
        }
        //if (thresholds[3] > 1000) {
        //    threshFilter->SetLowerThreshold(thresholds[2]);
        //}
        //else {
        //    threshFilter->SetLowerThreshold(thresholds[3]);
    
        threshFilter->SetLowerThreshold(thresholds[0]);
    }
    else {
        threshFilter->SetLowerThreshold(1);
    }
    threshFilter->SetUpperThreshold(10000);
    threshFilter->SetOutsideValue(0);
    threshFilter->SetInsideValue(1);
    threshFilter->Update();

    //using StructuringElementType = itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>;
    //StructuringElementType structuringElement;
    //structuringElement.SetRadius(1);
    //structuringElement.CreateStructuringElement();
    //
    //using BinaryMorphologicalClosingImageFilterType =
    //    itk::BinaryMorphologicalClosingImageFilter<ImageType, ImageType, StructuringElementType>;
    //BinaryMorphologicalClosingImageFilterType::Pointer closingFilter = BinaryMorphologicalClosingImageFilterType::New();
    //closingFilter->SetInput(threshFilter->GetOutput());
    //closingFilter->SetForegroundValue(1);
    //closingFilter->SetKernel(structuringElement);
    //closingFilter->Update();
    //image_to_process = closingFilter->GetOutput();

    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(threshFilter->GetOutput());
    duplicator->Update();

    //image_to_process = close_on_2d_slices(threshFilter->GetOutput(), 1, 1);
    image_to_process = threshFilter->GetOutput();

    std::string name_of_the_file;
    name_of_the_file = std::to_string(number_of_test_file) + "_threshold_ribs.nii.gz";


    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(image_to_process);
    try
    {
        if (!is_cleaned_from_non_ribs) {
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    ImageType::IndexType index3d;
    ImageType::SizeType size_ribs = image_to_process->GetLargestPossibleRegion().GetSize();


    //int offset_l_x = -70;
    //int offset_r_x = 70;

    int offset_l_x = -35;
    int offset_r_x = 35;

    int offset_down_y = 35;

    bool there_was_a_line_before = false;

    short pixelIntensity_l = 0;
    short pixelIntensity_r = 0;

    short actual_pixel_intesity = 0;

    int old_seed_z = 0;
    int old_seed_r_z = 0;

    int old_seed_y_l = 0;
    int old_seed_y_r = 0;

    int end = 80;

    int min_space_bw_ribs = 6;

    bool jump = false;

    int y_c = 0;
    int y_cc = 0;

    int width_of_the_mark = 20;

    ImageType::IndexType seed_l;
    ImageType::IndexType seed_r;


    // One by one

    std::vector<ImageType::IndexType> seed_vec_l;
    std::vector<ImageType::IndexType> seed_vec_r;

    for (int z = 0; z < size_ribs[2]; z++) {
        CirclesListType::const_iterator itCircles = circles_list[z].begin();


        int x_c = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0];


        if (old_seed_y_l == 0) {
            y_c = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1] - (*itCircles)->GetRadius()[0];
        }
        else {
            y_cc = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1];
        }

        offset_l_x = -(*itCircles)->GetRadius()[0] - 7;
        offset_r_x = (*itCircles)->GetRadius()[0] + 7;

        //offset_l_x = -(*itCircles)->GetRadius()[0];
        //offset_r_x = (*itCircles)->GetRadius()[0];

        for (int x = 0; x < width_of_the_mark; x++) {
            for (int y = 0; y < size_ribs[1]; y++) {


                jump = false;

                index3d[1] = y;
                index3d[2] = z;

                // LEFT
                index3d[0] = x_c + offset_l_x - x;
                if (x == width_of_the_mark - 1 && y_c + end > y && y > y_c) {
                    pixelIntensity_l += image_to_process->GetPixel(index3d);

                }
                actual_pixel_intesity = image_to_process->GetPixel(index3d);

                if (pixelIntensity_l != 0) {


                    // set the seed point
                    seed_l[0] = index3d[0] - 1;
                    seed_l[1] = index3d[1];
                    seed_l[2] = index3d[2];


                    if (image_to_process->GetPixel(seed_l) == 1 && x == 19) {
                        seed_l[0]++;
                        //neighborhoodConnected->AddSeed(seed_l);
                        seed_vec_l.push_back(seed_l);
                        // TODO: std::cout << "SEEDADDED2_l " << seed_l[2] << std::endl;
                        // TODO: std::cout << "SEEDADDED0_l " << seed_l[0] << std::endl;
                        // TODO: std::cout << "SEEDADDED1_l " << seed_l[1] << std::endl;
                        // TODO: std::cout << "-----------------" << std::endl;

                    }

                }
                if (actual_pixel_intesity != 0) {

                    image_to_process->SetPixel(index3d, -1);
                    duplicator->GetOutput()->SetPixel(index3d, -1);
                }
                else {
                    if (y_c + end > y && y > y_c) {
                        image_to_process->SetPixel(index3d, -2);
                        duplicator->GetOutput()->SetPixel(index3d, -2);
                    }
                    else {
                        image_to_process->SetPixel(index3d, -3);
                        duplicator->GetOutput()->SetPixel(index3d, -3);
                    }

                }
                
                actual_pixel_intesity = 0;
                pixelIntensity_l = 0;

                // RIGHT
                index3d[0] = x_c + offset_r_x + x;
                if (x == width_of_the_mark - 1 && y_c + end > y && y > y_c) {
                    pixelIntensity_r += image_to_process->GetPixel(index3d);
                }
                
                actual_pixel_intesity = image_to_process->GetPixel(index3d);
                if (pixelIntensity_r != 0) {
                
                
                    // set the seed point
                    seed_r[0] = index3d[0] + 1;
                    seed_r[1] = index3d[1];
                    seed_r[2] = index3d[2];
                
                
                    if (image_to_process->GetPixel(seed_r) == 1 && x == 19) {
                        seed_r[0]--;
                        // neighborhoodConnected->AddSeed(seed_r);
                        seed_vec_r.push_back(seed_r);
                        // TODO: std::cout << "SEEDADDED0_r " << seed_r[0] << std::endl;
                        // TODO: std::cout << "SEEDADDED1_r " << seed_r[1] << std::endl;
                        // TODO: std::cout << "SEEDADDED2_r " << seed_r[2] << std::endl;
                        // TODO: std::cout << "-----------------" << std::endl;
                
                    }
                }
                if (actual_pixel_intesity != 0) {
                
                    image_to_process->SetPixel(index3d, -1);
                    duplicator->GetOutput()->SetPixel(index3d, -1);
                }
                else {
                    if (y_c + end > y && y > y_c) {
                        image_to_process->SetPixel(index3d, -2);
                        duplicator->GetOutput()->SetPixel(index3d, -2);
                    }
                    else {
                        image_to_process->SetPixel(index3d, -3);
                        duplicator->GetOutput()->SetPixel(index3d, -3);
                    }
                }
                actual_pixel_intesity = 0;
                pixelIntensity_r = 0;


            }
            if (!is_cleaned_from_non_ribs) {
                // KEEP ALL
                for (int seed_ind = 0; seed_ind < seed_vec_l.size(); seed_ind++) {
                    neighborhoodConnected->AddSeed(seed_vec_l[seed_ind]);
                    seed_vec_global_l.push_back(seed_vec_l[seed_ind]);
                }
                seed_vec_l.clear();
                for (int seed_ind = 0; seed_ind < seed_vec_r.size(); seed_ind++) {
                    neighborhoodConnected->AddSeed(seed_vec_r[seed_ind]);
                    seed_vec_global_r.push_back(seed_vec_r[seed_ind]);
                }
                seed_vec_r.clear();
            }
            else {
                // KEEP ONLY THE FIRST
                if (!seed_vec_l.empty()) {
                    neighborhoodConnected->AddSeed(seed_vec_l[0]);
                    seed_vec_global_l.push_back(seed_vec_l[0]);
                    seed_vec_l.clear();
                }
                if (!seed_vec_r.empty()) {
                    neighborhoodConnected->AddSeed(seed_vec_r[0]);
                    seed_vec_global_r.push_back(seed_vec_r[0]);
                    seed_vec_r.clear();
                }
            }
        }

        pixelIntensity_l = 0;
        pixelIntensity_r = 0;
    }

    
    name_of_the_file = std::to_string(number_of_test_file) + "_threshold_ribs_mark.nii.gz";

    writerT->SetFileName(name_of_the_file);
    writerT->SetInput(image_to_process);
    try
    {
        if (!is_cleaned_from_non_ribs) {
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    //smoothing->SetInput(reader->GetOutput());

    // IMP
    neighborhoodConnected->SetInput(image_to_process);
    //neighborhoodConnected->SetInput(duplicator->GetOutput());

    //smoothing->SetNumberOfIterations(5);
    //smoothing->SetTimeStep(0.125);

    const PixelType lowerThreshold = 1;
    const PixelType upperThreshold = 1;

    neighborhoodConnected->SetLower(lowerThreshold);
    neighborhoodConnected->SetUpper(upperThreshold);

    ImageType::SizeType radius;
    radius[0] = 0;   // two pixels along x
    radius[1] = 0;   // two pixels along y
    radius[2] = 0;   // two pixels along z
    neighborhoodConnected->SetRadius(radius);
    
    //seed[0] = 160;
    //seed[1] = 425;
    //seed[2] = 148;
    //
    //neighborhoodConnected->SetSeed(seed);
    neighborhoodConnected->SetReplaceValue(20);

    neighborhoodConnected->Update();

    caster->SetInput(neighborhoodConnected->GetOutput());
    caster->Update();
    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow.nii.gz";

    
    try
    {
        if (!is_cleaned_from_non_ribs) {
            writerT->SetFileName(name_of_the_file);
            writerT->SetInput(caster->GetOutput());
            writerT->Update();
        }
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    return caster->GetOutput();

}

int count_reggrow_size(ImageType::Pointer ribs_only, ImageType::IndexType seed) {
    int ret_val = 0;

    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();

    const PixelType lowerThreshold = 20;
    const PixelType upperThreshold = 20;

    neighborhoodConnected->SetLower(lowerThreshold);
    neighborhoodConnected->SetUpper(upperThreshold);
    ImageType::SizeType radius;
    radius[0] = 0;   // two pixels along x
    radius[1] = 0;   // two pixels along y
    radius[2] = 0;   // two pixels along z
    neighborhoodConnected->SetRadius(radius);
    neighborhoodConnected->SetInput(ribs_only);
    neighborhoodConnected->AddSeed(seed);
    neighborhoodConnected->SetReplaceValue(1);
    neighborhoodConnected->Update();


    ImageType::IndexType index3d;
    ImageType::SizeType size = ribs_only->GetLargestPossibleRegion().GetSize();

    short pixelIntensity = 0;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                index3d[0] = x;
                index3d[1] = y;
                index3d[2] = z;
                pixelIntensity = neighborhoodConnected->GetOutput()->GetPixel(index3d);
                
                ret_val += pixelIntensity;

            }
        }
    }

    //std::cout << "SEED: " << seed[0] << " " << seed[1] << " " << seed[2] << std::endl;
    //std::cout << "retv: " << ret_val << std::endl;


    return ret_val;
}

int count_non_zero_vox(ImageType::Pointer img) {
    int ret_val = 0;


    ImageType::IndexType index3d;
    ImageType::SizeType size = img->GetLargestPossibleRegion().GetSize();

    short pixelIntensity = 0;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                index3d[0] = x;
                index3d[1] = y;
                index3d[2] = z;
                pixelIntensity = img->GetPixel(index3d);
                if (pixelIntensity > 0) {
                    ret_val++;
                }
            }
        }
    }


    return ret_val;
}

ImageType::IndexType get_optimal_seed(ImageType::Pointer ribs_only, std::vector<ImageType::IndexType> seed_vec, int begin, int end) {
    ImageType::IndexType ret_val;
    ImageType::IndexType temp;
    //ret_val[0] = 217;
    //ret_val[1] = 313;
    //ret_val[2] = 8;

    int max = -99999999;


    for (int i = begin; i <= end; i++) {
        temp = seed_vec[i];
        //std::cout << temp[0] << " ";
        //std::cout << temp[1] << " ";
        //std::cout << temp[2] << " ";
        //std::cout << "-----------------\n";

    }
    //std::cout << "======================\n";
   
    for (int i = begin; i <= end; i++) {
        temp = seed_vec[i];
        int reg_grow_volume = count_reggrow_size(ribs_only, temp);
        if (reg_grow_volume >= max) {
            max = reg_grow_volume;
            ret_val[0] = temp[0];
            ret_val[1] = temp[1];
            ret_val[2] = temp[2];
        }
    }

    return ret_val;
}

bool check_if_lapocka(std::vector<ImageType::IndexType> seed_vec, int begin, int end) {
    bool is_lapocka = false;
    if (begin - 1 < 0 || end + 1 >= seed_vec.size()) {
        return is_lapocka;
    }
    else {
        int before_z = seed_vec[begin - 1][2];
        int after_z = seed_vec[end + 1][2];

        int begin_z = seed_vec[begin][2];
        int end_z = seed_vec[end][2];

        if (before_z + 1 == begin_z && after_z - 1 == end_z) {
            is_lapocka = true;
        }
    }

    return is_lapocka;
}

// REMOVE NON RIBS
ImageType::Pointer post_process(ImageType::Pointer input) {

    ImageType::IndexType index3d;
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();

    short pixelIntensity = 0;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                index3d[0] = x;
                index3d[1] = y;
                index3d[2] = z;
                pixelIntensity = input->GetPixel(index3d);

                if (pixelIntensity > 24) {
                    input->SetPixel(index3d, 0);
                }

            }
        }
    }

    return input;
}

/**
 * Adds img1 and img2 and returns the result
 */
ImageType::Pointer add_images(ImageType::Pointer img1, ImageType::Pointer img2) {

    ImageType::IndexType index3d;
    ImageType::SizeType size = img1->GetLargestPossibleRegion().GetSize();

    short pixelIntensity_img1 = 0;
    short pixelIntensity_img2 = 0;
    for (int z = 0; z < size[2]; z++) {
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                index3d[0] = x;
                index3d[1] = y;
                index3d[2] = z;
                pixelIntensity_img1 = img1->GetPixel(index3d);
                pixelIntensity_img2 = img2->GetPixel(index3d);

                if (pixelIntensity_img1 == 0) {
                    img1->SetPixel(index3d, pixelIntensity_img2);
                }
            }
        }
    }

    return img1;
}

/**
 * Draws a line on img
 * 
 * @param img iamge to draw a line
 * @param index z index of the line
 * @param circles_list list of the circles
 */
void draw_line_for_vertebra(ImageType::Pointer img, ImageType::IndexType index, std::vector<CirclesListType> circles_list) {
    ImageType::IndexType index3d;
    ImageType::SizeType size = img->GetLargestPossibleRegion().GetSize();
    if (size[2] <= index[2]) {
        std::cout << "BAD VERTEBRA LINE" << std::endl;
        return;
    }
    
    index3d[2] = index[2];

    short pixelIntensity = 1;
    CirclesListType::const_iterator itCircles = circles_list[index[2]].begin();
    
    int x = (*itCircles)->GetObjectToParentTransform()->GetOffset()[0] - 40;
    
    int x_end = x + 80;

    for (; x < x_end; x++) {

        int y = (*itCircles)->GetObjectToParentTransform()->GetOffset()[1] - 40;
        int y_end = y + 80;

        for (; y < y_end; y++) {
            index3d[0] = x;
            index3d[1] = y;
            img->SetPixel(index3d, pixelIntensity);
        }
    }
}

/**
 * Labels ribs by anathomical oreder based on the seed points and draws line betweem vertebras
 *
 * @param ribs_only image to bel labeled
 * @param ribs_only_labeled image with the old non anathomical labels
 * @param number_of_test_file 
 * @param number_of_test_file key number of the image
 * @param circles_list list of the circles for line drawing
 */

void label_ribs(ImageType::Pointer ribs_only, ImageType::Pointer ribs_only_labeled, int number_of_test_file, std::vector<CirclesListType> circles_list) {
    //ribs_only = open_on_2d_slices(ribs_only, 1, 20);
    std::vector<ImageType::IndexType> optimal_seed_vec_l;
    std::vector<ImageType::IndexType> optimal_seed_vec_r;

    std::vector<ImageType::IndexType> vertebra_lines_vec_l;
    std::vector<ImageType::IndexType> vertebra_lines_vec_r;

    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    ConnectedFilterType::Pointer neighborhoodConnected = ConnectedFilterType::New();

    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    // addFilter->SetInput1(image1);
    // addFilter->SetInput2(image2);
    // addFilter->Update();

    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(ribs_only);
    duplicator->Update();

    ImageType::Pointer ribs_labeled = duplicator->GetOutput();
    ribs_labeled->FillBuffer(0);

    duplicator->SetInputImage(ribs_labeled);
    duplicator->Update();

    ImageType::Pointer vertebra_lines = duplicator->GetOutput();


    const PixelType lowerThreshold = 20;
    const PixelType upperThreshold = 20;

    neighborhoodConnected->SetLower(lowerThreshold);
    neighborhoodConnected->SetUpper(upperThreshold);

    ImageType::SizeType radius;
    radius[0] = 0;
    radius[1] = 0;
    radius[2] = 0;
    neighborhoodConnected->SetRadius(radius);

    // TEST
    std::vector<ImageType::IndexType> test_seeds;


    ImageType::IndexType seed1_test;
    seed1_test[0] = 217;
    seed1_test[1] = 313;
    seed1_test[2] = 8;

    ImageType::IndexType seed2_test;
    seed2_test[0] = 215;
    seed2_test[1] = 322;
    seed2_test[2] = 22;

    test_seeds.push_back(seed1_test);
    test_seeds.push_back(seed2_test);

    neighborhoodConnected->SetInput(ribs_only);

    addFilter->SetInput1(ribs_labeled);
    addFilter->SetInput2(ribs_labeled);
    addFilter->Update();

    int is_lapocka = false;
    int ind_l = 1;
    int ind_r = 2;
    int min_size_of_rib = 10;

    std::set<int> added_label_values;

    for (int i = 0; i < seed_vec_global_l.size(); i++) {
        int j = 0;
        while (i + j < seed_vec_global_l.size() - 1 && ((seed_vec_global_l[i + j + 1].GetIndex()[2] - seed_vec_global_l[i + j].GetIndex()[2]) == 1) && std::abs(seed_vec_global_l[i + j + 1].GetIndex()[1] - seed_vec_global_l[i + j].GetIndex()[1]) < 19) {
        //while (i + j < seed_vec_global_l.size() - 1 && ((seed_vec_global_l[i + j + 1].GetIndex()[2] - seed_vec_global_l[i + j].GetIndex()[2]) == 1)) {
            j++;
        }
        //seed_vec_global_l.erase(seed_vec_global_l.begin() + i + 1, seed_vec_global_l.begin() + i + j);

        //is_lapocka = check_if_lapocka(seed_vec_global_l, i, i + j);

        int begin = i;
        int end = i + j;

       
        //if (!is_lapocka) {
        //optimal_seed_vec_l.push_back(get_optimal_seed(ribs_only, seed_vec_global_l, i, i + j));
        
        // Draw vertebra line
        //draw_line_for_vertebra(vertebra_lines, seed_vec_global_l[begin]);
        //draw_line_for_vertebra(vertebra_lines, seed_vec_global_l[end]);
        vertebra_lines_vec_l.push_back(seed_vec_global_l[end]);

        for (int seed_i = begin; seed_i <= end; seed_i++) {
            //for (ImageType::IndexType index3d_temp_l : test_seeds) {
            //std::cout << seed_vec_global_l[seed_i] << "--------+" << std::endl;
            //std::cout << "i: " << ind_l << std::endl;
            neighborhoodConnected->AddSeed(seed_vec_global_l[seed_i]);
        
        }

        neighborhoodConnected->AddSeed(seed_vec_global_l[begin]);
            
        //std::cout << "======================+\n";

        //if (count_non_zero_vox(neighborhoodConnected->GetOutput()) > min_size_of_rib) {
        if (added_label_values.find(ribs_only_labeled->GetPixel(seed_vec_global_l[begin])) == added_label_values.end()) {   
            neighborhoodConnected->SetReplaceValue(ind_l);
            neighborhoodConnected->Update();

            ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
            //ribs_labeled = add_images(ribs_labeled, close_on_2d_slices(neighborhoodConnected->GetOutput(), 2, ind_l));
            ind_l += 2;
            added_label_values.insert(ribs_only_labeled->GetPixel(seed_vec_global_l[begin]));
        }
        neighborhoodConnected->ClearSeeds();
        //}
        i = i + j;
    }
    neighborhoodConnected->ClearSeeds();
    for (int i = 0; i < seed_vec_global_r.size(); i++) {
        int j = 0;
        while (i + j < seed_vec_global_r.size() - 1 && ((seed_vec_global_r[i + j + 1].GetIndex()[2] - seed_vec_global_r[i + j].GetIndex()[2]) == 1) && std::abs(seed_vec_global_r[i + j + 1].GetIndex()[1] - seed_vec_global_r[i + j].GetIndex()[1]) < 19) {
        //while (i + j < seed_vec_global_r.size() - 1 && ((seed_vec_global_r[i + j + 1].GetIndex()[2] - seed_vec_global_r[i + j].GetIndex()[2]) == 1)) {
            j++;
        }
        //seed_vec_global_r.erase(seed_vec_global_r.begin() + i + 1, seed_vec_global_r.begin() + i + j);

        //is_lapocka = check_if_lapocka(seed_vec_global_r, i, i + j);

        int begin = i;
        int end = i + j;

        // Draw vertebra line
        //draw_line_for_vertebra(vertebra_lines, seed_vec_global_r[begin]);
        //draw_line_for_vertebra(vertebra_lines, seed_vec_global_r[end]);
        vertebra_lines_vec_r.push_back(seed_vec_global_r[end]);
        //if (!is_lapocka) {
        //optimal_seed_vec_r.push_back(get_optimal_seed(ribs_only, seed_vec_global_r, i, i + j));
            
        for (int seed_i = begin; seed_i <= end; seed_i++) {
            for (ImageType::IndexType index3d_temp_l : test_seeds) {
                std::cout << seed_vec_global_r[seed_i] << "--------+" << std::endl;
                std::cout << "i: " << ind_r << std::endl;
            }
            neighborhoodConnected->AddSeed(seed_vec_global_r[seed_i]);
        
        }

        neighborhoodConnected->AddSeed(seed_vec_global_r[begin]);

        std::cout << "======================+\n";
        //if (count_non_zero_vox(neighborhoodConnected->GetOutput()) > min_size_of_rib) {
        if (added_label_values.find(ribs_only_labeled->GetPixel(seed_vec_global_r[begin])) == added_label_values.end()) {
            neighborhoodConnected->SetReplaceValue(ind_r);
            neighborhoodConnected->Update();

            ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
            //ribs_labeled = add_images(ribs_labeled, close_on_2d_slices(neighborhoodConnected->GetOutput(), 2, ind_r));
            ind_r += 2;
            added_label_values.insert(ribs_only_labeled->GetPixel(seed_vec_global_r[begin]));
        }
        neighborhoodConnected->ClearSeeds();
        //}
        i = i + j;
    }

    for (int i = 0; i < vertebra_lines_vec_l.size(); i++) {
         
        int avg = std::ceil((vertebra_lines_vec_l[i][2] + vertebra_lines_vec_r[i][2])/2);
        ImageType::IndexType index;
        index[2] = avg;

        draw_line_for_vertebra(vertebra_lines, index, circles_list);
    }


    // // OLD
    // //for (ImageType::IndexType index3d_temp_l : seed_vec_global_l) {
    // for (ImageType::IndexType index3d_temp_l : optimal_seed_vec_l) {
    // //for (ImageType::IndexType index3d_temp_l : test_seeds) {
    //     std::cout << index3d_temp_l << std::endl;
    //     neighborhoodConnected->ClearSeeds();
    //     std::cout << "i: " << ind_l << std::endl;
    //     neighborhoodConnected->AddSeed(index3d_temp_l);
    //     neighborhoodConnected->SetReplaceValue(ind_l);
    //     neighborhoodConnected->Update();
    // 
    //     ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
    // 
    //     //addFilter->SetInput1(addFilter->GetOutput());
    //     //addFilter->SetInput2(neighborhoodConnected->GetOutput());
    //     //addFilter->Update();
    //     ind_l += 2;
    // }
    // 
    // 
    // //for (ImageType::IndexType index3d_temp_r : seed_vec_global_r) {
    // for (ImageType::IndexType index3d_temp_r : optimal_seed_vec_r) {
    //     //for (ImageType::IndexType index3d_temp_l : test_seeds) {
    //     std::cout << index3d_temp_r << std::endl;
    //     neighborhoodConnected->ClearSeeds();
    //     std::cout << "i: " << ind_r << std::endl;
    //     neighborhoodConnected->AddSeed(index3d_temp_r);
    //     neighborhoodConnected->SetReplaceValue(ind_r);
    //     neighborhoodConnected->Update();
    // 
    //     ribs_labeled = add_images(ribs_labeled, neighborhoodConnected->GetOutput());
    // 
    //     //addFilter->SetInput1(addFilter->GetOutput());
    //     //addFilter->SetInput2(neighborhoodConnected->GetOutput());
    //     //addFilter->Update();
    //     ind_r += 2;
    // }

    //seed[0] = 1;
    //seed[1] = 425;
    //seed[2] = 148;
    //
    //neighborhoodConnected->SetSeed(seed);
    //neighborhoodConnected->SetReplaceValue(8);

    //neighborhoodConnected->Update();

    typedef itk::CastImageFilter< ImageType, ImageType >
        CastingFilterType;
    CastingFilterType::Pointer caster = CastingFilterType::New();

    caster->SetInput(neighborhoodConnected->GetOutput());

    std::string name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_mylabel.nii.gz";

    writerT->SetFileName(name_of_the_file);
    //writerT->SetInput(addFilter->GetOutput());
    writerT->SetInput(ribs_labeled);
    //writerT->SetInput(post_process(addFilter->GetOutput()));
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }
    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_vert_line.nii.gz";
    writerT->SetFileName(name_of_the_file);
    //writerT->SetInput(addFilter->GetOutput());
    writerT->SetInput(vertebra_lines);
    //writerT->SetInput(post_process(addFilter->GetOutput()));
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }
}

/**
* Calculates and normalizes the image's histogram
*
* @param input image 
*
* @return normalized histogram
*/
ScalarImageToHistogramGeneratorType::Pointer normalize(ImageType::Pointer input) {

    ScalarImageToHistogramGeneratorType::Pointer scalarImageToHistogramGenerator = ScalarImageToHistogramGeneratorType::New();

    scalarImageToHistogramGenerator->SetHistogramMin(0);
    scalarImageToHistogramGenerator->SetHistogramMax(1000);
    scalarImageToHistogramGenerator->SetNumberOfBins(50);
    scalarImageToHistogramGenerator->SetAutoHistogramMinimumMaximum(false);
    scalarImageToHistogramGenerator->SetInput(input);
    scalarImageToHistogramGenerator->Compute();

    return scalarImageToHistogramGenerator;

    // ImageType::IndexType index3d;
    // for (int x = 0; x < input->GetOutput()->GetBufferedRegion().GetSize()[0]; x++) {
    //     for (int y = 0; y < input->GetOutput()->GetBufferedRegion().GetSize()[1]; y++) {
    //         for (int z = 0; z < input->GetOutput()->GetBufferedRegion().GetSize()[2]; z++) {
    //             index3d[0] = x;
    //             index3d[1] = y;
    //             index3d[2] = z;
    // 
    //             PixelType pix = input->GetOutput()->GetPixel(index3d);
    // 
    //             if (pix < -1000) {
    //                 input->GetOutput()->SetPixel(index3d, -1000);
    //             }
    //             if (pix > 1000) {
    //                 input->GetOutput()->SetPixel(index3d, 1000);
    //             }
    //         }
    //     }
    // }
}

/**
* Otsu threshold filter based on the image histogram
*
* @param input histogram to calculate the threshold
*
* @return thresholds provided by otsu
*/
CalculatorType::OutputType otusHist(ScalarImageToHistogramGeneratorType::Pointer input) {

    //OtsuFilterType::Pointer otsuFilter = OtsuFilterType::New();
    CalculatorType::Pointer otsuFilter = CalculatorType::New();
    //otsuFilter->SetInput(reader_original->GetOutput());
    otsuFilter->SetInputHistogram(input->GetOutput());
    otsuFilter->SetNumberOfThresholds(1);
    otsuFilter->Update(); // To compute threshold

                          //OtsuFilterType::ThresholdVectorType thresholds = otsuFilter->GetThresholds();
    const CalculatorType::OutputType & thresholds = otsuFilter->GetOutput();
    return thresholds;
}

/**
 * Otsu threshold filter based on the image
 *
 * @param input image to be thresholded
 *
 * @return thresholds provided by otsu
 */
OtsuFilterType::ThresholdVectorType otusImg(ReaderType::Pointer input) {
    
    OtsuFilterType::Pointer otsuFilter = OtsuFilterType::New();
    otsuFilter->SetInput(input->GetOutput());
    otsuFilter->SetNumberOfThresholds(1);
    otsuFilter->Update(); // To compute threshold

    OtsuFilterType::ThresholdVectorType thresholds = otsuFilter->GetThresholds();
    return thresholds;
}

/**
* Writes out the image's histogram to a .txt file
*
* @param histogram histogram to be written
* @param number_of_test_file key number of the image
*/
void write_hist(ScalarImageToHistogramGeneratorType::Pointer histogram, int number_of_test_file) {
    std::ofstream myfile;
    myfile.open(std::to_string(number_of_test_file) + "_norm_hist.txt");
    myfile << "[ ";
    for (int i = 0; i < 50; i++) {
        myfile << histogram->GetOutput()->GetFrequency(i) << " ";
        //std::cout << a->GetOutput()->GetFrequency(0) ;

    }
    myfile << "]";
    myfile.close();
}


struct Label_Princ {
    int label;
    double princ_mom;
};

struct by_princ_mom {
    bool operator()(Label_Princ const &a, Label_Princ const &b) const noexcept {
        return a.princ_mom < b.princ_mom;
    }
};

/** 
 * Labels the image containing the ribs based on its connectivity, and removes non rib objects based on their attributes
 *
 * @param ribs_only segmented images contains the result of the reginog growing
 * @param number_of_test_file key number of the image
 *
 * @return labeled image by conncentedCompontes algorithm
 */
ImageType::Pointer label_ribs_ccomponents(ImageType::Pointer ribs_only, int number_of_test_file) {

    ImageType::Pointer ret_val;

    ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New();
    //connected->FullyConnectedOn();
    connected->SetInput(ribs_only);
    connected->Update();

    std::cout << "Number of objects: " << connected->GetObjectCount() << std::endl;

    std::string name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_labed_cc.nii.gz";
    using WriterTypeT = itk::ImageFileWriter< ImageType >;
    WriterTypeT::Pointer writerT = WriterTypeT::New();
    writerT->SetFileName(name_of_the_file);
    //writerT->SetInput(addFilter->GetOutput());
    writerT->SetInput(connected->GetOutput());
    //writerT->SetInput(post_process(addFilter->GetOutput()));
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    I2LType::Pointer i2l = I2LType::New();
    i2l->SetInput(connected->GetOutput());
    i2l->SetComputePerimeter(false);
    i2l->Update();

    LabelMapType * labelMap = i2l->GetOutput();

    int labelNumberBeforeFilters = labelMap->GetNumberOfLabelObjects();

    std::vector<int> remove_vec;
    std::vector<Label_Princ> Label_Princ_vec;

    // Retrieve all attributes
    for (unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); n++)
    {
        std::cout << "---------------------" << std::endl;
        ShapeLabelObjectType * labelObject = labelMap->GetNthLabelObject(n);
        int label_no = itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel());

        //if (label_no ==7 || label_no == 29 || label_no == 27) 
        {

            std::cout << "Label: " << label_no << std::endl;
            //std::cout << "    BoundingBox: " << labelObject->GetBoundingBox() << std::endl;
            std::cout << "    NumberOfPixels: " << labelObject->GetNumberOfPixels() << std::endl;
            // std::cout << "    PhysicalSize: " << labelObject->GetPhysicalSize() << std::endl;
            std::cout << "    Centroid: " << labelObject->GetCentroid() << std::endl;
            // std::cout << "    NumberOfPixelsOnBorder: " << labelObject->GetNumberOfPixelsOnBorder() << std::endl;
            // std::cout << "    PerimeterOnBorder: " << labelObject->GetPerimeterOnBorder() << std::endl;
            // std::cout << "    FeretDiameter: " << labelObject->GetFeretDiameter() << std::endl;
            std::cout << "    PrincipalMoments: " << labelObject->GetPrincipalMoments() << std::endl;
            //std::cout << "    PrincipalAxes: " << labelObject->GetPrincipalAxes() << std::endl;
            std::cout << "    Elongation: " << labelObject->GetElongation() << std::endl;
            // std::cout << "    Perimeter: " << labelObject->GetPerimeter() << std::endl;
            // std::cout << "    Roundness: " << labelObject->GetRoundness() << std::endl;
            // std::cout << "    EquivalentSphericalRadius: " << labelObject->GetEquivalentSphericalRadius() << std::endl;
            // std::cout << "    EquivalentSphericalPerimeter: " << labelObject->GetEquivalentSphericalPerimeter() << std::endl;
            // std::cout << "    EquivalentEllipsoidDiameter: " << labelObject->GetEquivalentEllipsoidDiameter() << std::endl;
            // std::cout << "    Flatness: " << labelObject->GetFlatness() << std::endl;
            // std::cout << "    PerimeterOnBorderRatio: " << labelObject->GetPerimeterOnBorderRatio() << std::endl;

        }
        double val = labelObject->GetPrincipalMoments()[0];

        //if (!(0.5 < labelObject->GetPrincipalMoments()[0] && labelObject->GetPrincipalMoments()[0] < 30)) {
        if (!(labelObject->GetPrincipalMoments()[0] < 50)) {
        //if (label_no != 0 && labelObject->GetElongation() < 3) {
            remove_vec.push_back(label_no);
            std::cout << "The label " << label_no << " has been removed" << std::endl;
            std::cout << "++++++++++++++++++++" << std::endl;
        } else {
           std::cout << "---------------------"<<std::endl;
        }
    }

    for (int temp : remove_vec) {
        std::cout << " LABEL: " << temp << std::endl;
        labelMap->RemoveLabel(temp);
    }

    remove_vec.clear();

    LabelMapToLabelImageFilterType::Pointer labelImageConverter = LabelMapToLabelImageFilterType::New();
    labelImageConverter->SetInput(labelMap);

    int num_of_ribs = 24;

    std::cout << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cout << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
    std::cout << "Num of labels after filter " << labelMap->GetNumberOfLabelObjects() << std::endl;

    std::cerr << "---------------------" << std::endl;
    std::cerr << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cerr << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
    std::cerr << "Num of labels after filter " << labelMap->GetNumberOfLabelObjects() << std::endl;
    std::cerr << "---------------------" << std::endl;

    if (labelMap->GetNumberOfLabelObjects() > num_of_ribs) {
        
        std::cerr << "To much objects there!!!" << std::endl;

        for (unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); n++)
        {
            ShapeLabelObjectType * labelObject = labelMap->GetNthLabelObject(n);
            int label_no = itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel());
            //std::cout << "Label: " << label_no << std::endl;
            
            Label_Princ temp;
            temp.label = label_no;
            temp.princ_mom = labelObject->GetPrincipalMoments()[0];

            Label_Princ_vec.push_back(temp);
        }
        int num_of_labels = labelMap->GetNumberOfLabelObjects();
        std::sort(Label_Princ_vec.begin(), Label_Princ_vec.end(), by_princ_mom());

        for (int i = 0; i < num_of_labels - num_of_ribs; i++) {
            std::cout << " LABEL: " << Label_Princ_vec[i].label << std::endl;
            labelMap->RemoveLabel(Label_Princ_vec[i].label);
        }
        labelMap->Update();
        LabelMapToLabelImageFilterType::Pointer labelImageConverter = LabelMapToLabelImageFilterType::New();
        labelImageConverter->SetInput(labelMap);
        labelImageConverter->Update();
        writerT->SetInput(labelImageConverter->GetOutput());
        ret_val = labelImageConverter->GetOutput();

       // LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter =
       //     LabelShapeKeepNObjectsImageFilterType::New();
       // labelShapeKeepNObjectsImageFilter->SetInput(labelImageConverter->GetOutput());
       // labelShapeKeepNObjectsImageFilter->SetBackgroundValue(0);
       // labelShapeKeepNObjectsImageFilter->SetNumberOfObjects(num_of_ribs);
       // labelShapeKeepNObjectsImageFilter->ReverseOrderingOff();
       //
       // labelShapeKeepNObjectsImageFilter->SetAttribute(
       //     //LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
       //     //LabelShapeKeepNObjectsImageFilterType::LabelObjectType::BOUNDING_BOX);
       //     LabelShapeKeepNObjectsImageFilterType::LabelObjectType::ELONGATION);
       //     //LabelShapeKeepNObjectsImageFilterType::LabelObjectType::PRINCIPAL_MOMENTS);
       // labelShapeKeepNObjectsImageFilter->Update();
       // 
       // writerT->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());
       // ret_val = labelShapeKeepNObjectsImageFilter->GetOutput();
        


        //std::cout << "Objects removed: " << labelNumberBeforeFilters - labelShapeKeepNObjectsImageFilter->GetNumberOfObjects() << std::endl;
        std::cout << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cout << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
        std::cout << "Num of labels after filter FINAL " << labelMap->GetNumberOfLabelObjects() << std::endl;

        std::cerr << "---------------------" << std::endl;
        std::cerr << "Objects removed: " << labelNumberBeforeFilters - labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cerr << "Num of labels before filter " << labelNumberBeforeFilters << std::endl;
        std::cerr << "Num of labels after filter FINAL " << labelMap->GetNumberOfLabelObjects() << std::endl;
        std::cerr << "---------------------" << std::endl;


    }
    else {
        writerT->SetInput(labelImageConverter->GetOutput());
        ret_val = labelImageConverter->GetOutput();
    
    }

    name_of_the_file = std::to_string(number_of_test_file) + "_reg_grow_labed_cc_N.nii.gz";

    writerT->SetFileName(name_of_the_file);
    //writerT->SetInput(addFilter->GetOutput());
    
    //writerT->SetInput(post_process(addFilter->GetOutput()));
    try
    {
        writerT->Update();
    }
    catch (itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    return ret_val;
}

/*
-20-
-21-
-23--
-28*
-31*
-47--
-49--
-52--
-54*
-57-
-58--

-55
-24

*/

//std::string files[] = { "test\\0verse112.nii.gz", "test\\1AnyScan.nii.gz", "test\\2LA_LD_02.nii.gz", "test\\3LA_1.nii.gz", "test\\4LUNG_PET.nii.gz", "test\\5LungRT.nii.gz", "test\\6LA_Diagn_1.nii.gz", "test\\7LA_Diagn_3.nii.gz", "test\\8LA_LD_01.nii.gz", "test\\9LungPhilips.nii.gz" };
//std::string files[] = { "test\\2LA_LD_02.nii.gz" };
//std::string files[] = { "test\\6LA_Diagn_1.nii.gz" };
//std::string files[] = { "test\\9LungPhilips.nii.gz" };
//std::string files[] = { "bigtest\\lung_08.nii.gz" };

//std::string files[] = { "bigtest\\lung_19.nii.gz" };
 // "bigtest\\lung_28.nii.gz","bigtest\\lung_31.nii.gz","bigtest\\lung_47.nii.gz",
 // "bigtest\\lung_49.nii.gz", "bigtest\\lung_52.nii.gz", "bigtest\\lung_54.nii.gz",
 // "bigtest\\lung_57.nii.gz", "bigtest\\lung_58.nii.gz"};

//std::string files[] = { "bigtest\\lung_20.nii.gz", "bigtest\\lung_21.nii.gz", "bigtest\\lung_57.nii.gz" };
//std::string files[] = { "bigtest\\lung_02.nii.gz"};
//std::string files[] = { "bigtest\\lung_12.nii.gz"};

//std::string files[] = { "bigtest\\lung_20.nii.gz", "bigtest\\lung_14.nii.gz"};

// TEST labeling on this

//std::string files[] = { "bigtest\\lung_08.nii.gz" };
//std::string files[] = { "bigtest\\lung_30.nii.gz" };
//std::string files[] = { "bigtest\\lung_01.nii.gz" };
//std::string files[] = { "bigtest\\lung_03.nii.gz" };
//std::string files[] = { "bigtest\\lung_01.nii.gz" };
//std::string files[] = { "bigtest\\lung_29.nii.gz" };
//std::string files[] = { "bigtest\\lung_33.nii.gz" };

//std::string files[] = { "bigtest\\lung_28.nii.gz" };
//std::string files[] = { "bigtest\\lung_49.nii.gz" };


//std::string files[] = { "bigtest\\lung_34.nii.gz" };

//std::string files[] = { "bigtest\\lung_33.nii.gz" };

//std::string files[] = { "bigtest\\lung_03.nii.gz" };


//std::string files[] = { "bigtest\\lung_07.nii.gz", "bigtest\\lung_10.nii.gz", "bigtest\\lung_12.nii.gz"};

//std::string files[] = { "bigtest\\lung_53.nii.gz"};
//std::string files[] = { "bigtest\\lung_60.nii.gz"};
//std::string files[] = { "bigtest\\lung_24.nii.gz"};
//std::string files[] = { "bigtest\\lung_30.nii.gz"};
//std::string files[] = { "bigtest\\lung_40.nii.gz"};
//std::string files[] = { "bigtest\\lung_44.nii.gz"};


//std::string files[] = { "bigtest\\lung_12.nii.gz"};
//std::string files[] = { "bigtest\\lung_46.nii.gz"};
//std::string files[] = { "bigtest\\lung_39.nii.gz"};
//std::string files[] = { "bigtest\\lung_26.nii.gz"};
//std::string files[] = { "bigtest\\lung_14.nii.gz"};
//std::string files[] = { "bigtest\\lung_13.nii.gz"};
//std::string files[] = { "bigtest\\lung_09.nii.gz"};


//std::string files[] = { "bigtest\\lung_27.nii.gz"};


//std::string files[] = { "slice_diff\\lung_01.nii.gz", "slice_diff\\lung_02.nii.gz" };

int main(int argc, char ** argv)
{
    int start_from_this_file = 1;


   std::vector<std::string> files;
   for (int i = start_from_this_file; i <= 60; i++) {
       if (i > 9) {
           files.push_back("bigtest\\lung_" + std::to_string(i) + ".nii.gz");
   
       }
       else {
           files.push_back("bigtest\\lung_0" + std::to_string(i) + ".nii.gz");
       }
   }


    if (argv[1] != nullptr) {
        files.clear();
        files.push_back(argv[1]);
    }

    int number_of_test_file = start_from_this_file;
    for (std::string file : files) {
        if ("test\\8LA_LD_01.nii.gz" == file) {
            number_of_test_file++;
            continue;
        }

        // LOG
        std::cerr << "FILE_no: " << number_of_test_file << std::endl;
        //-------------
        std::ofstream out(std::to_string(number_of_test_file) + "_zLOG.txt");
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
        //-------------

        using WriterTypeT = itk::ImageFileWriter< ImageType >;
        WriterTypeT::Pointer writerT = WriterTypeT::New();


        //const char * original_image_file_name = argv[1];
        std::string original_image_file_name = file;

        ReaderType::Pointer reader_original = ReaderType::New();
        ReaderType::Pointer reader_ribs = ReaderType::New();
        reader_original->SetFileName(original_image_file_name);
        reader_ribs->SetFileName(original_image_file_name);
        try
        {
            reader_original->Update(); // Image for spine extraction without corrigation
            reader_ribs->Update(); // Image for rib labeling
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        ReaderType::Pointer reader_original_1 = ReaderType::New();
        reader_original_1->SetFileName(original_image_file_name);
        try
        {
            reader_original_1->Update(); // Image for spine extraction with 1st corrigation
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        ReaderType::Pointer reader_original_2 = ReaderType::New();
        reader_original_2->SetFileName(original_image_file_name);
        try
        {
            reader_original_2->Update(); // // Image for spine extraction with 2nd corrigation
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        // Normalization of the image histogram for Otsu Threshold filter
        ScalarImageToHistogramGeneratorType::Pointer hist_of_original_img = normalize(reader_original->GetOutput());

        write_hist(hist_of_original_img, number_of_test_file);

        std::string name_of_the_file1 = std::to_string(number_of_test_file) + "_normalized.nii.gz";

        writerT->SetFileName(name_of_the_file1);
        writerT->SetInput(reader_ribs->GetOutput());
        try
        {
            writerT->Update();
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }



        // set up the extraction region [one slice]

        //const ImageType * inputImage = reader->GetOutput();

        // Thresholding the image based on the histogram normalized before *START*
        std::cout << "Thresholding" << std::endl;

        const CalculatorType::OutputType & thresholds = otusHist(hist_of_original_img);
        for (unsigned int i = 0; i < thresholds.size(); i++)
        {
            std::cout << thresholds[i] << std::endl;
        }

        using FilterType = itk::BinaryThresholdImageFilter< ImageType, ImageType >;
        FilterType::Pointer threshFilter = FilterType::New();
        threshFilter->SetInput(reader_original->GetOutput());
        // if (thresholds[3] > 1000) {
        //     threshFilter->SetLowerThreshold(thresholds[2]);
        // }
        // else {
        //     threshFilter->SetLowerThreshold(thresholds[3]);
        // }
        threshFilter->SetLowerThreshold(thresholds[0]);
        threshFilter->SetUpperThreshold(10000);
        threshFilter->SetOutsideValue(0);
        threshFilter->SetInsideValue(1);
        threshFilter->Update();
        // Thresholding the image based on the histogram normalized before *END*

        // CLOSING (for circles)
        //const ImageType::Pointer inputImage = close_on_2d_slices(threshFilter->GetOutput(), 1, 1);
        
        // Calculating the circles on the axial slices *circles_list* will conatin the lists of circles for axial slices *START*
        const ImageType * inputImage = threshFilter->GetOutput();

        ImageType::RegionType inputRegion = inputImage->GetBufferedRegion();
        ImageType::SizeType size = inputRegion.GetSize();
        int height_of_the_image;
        height_of_the_image = size[2];
        using ExtractFilterType = itk::ExtractImageFilter< ImageType, ImageType2D >;
        std::list<ImageType2D::Pointer> slice_list;
        std::vector<CirclesListType> circles_list;
        for (int i = 0; i < height_of_the_image; i++) {

            ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
            extractFilter->SetDirectionCollapseToSubmatrix();


            size[2] = 0; // we extract along z direction

            ImageType::IndexType start = inputRegion.GetIndex();

            //const unsigned int sliceNumber = std::stoi( argv[3] );

            start[2] = i;
            ImageType::RegionType desiredRegion;
            desiredRegion.SetSize(size);
            desiredRegion.SetIndex(start);

            extractFilter->SetExtractionRegion(desiredRegion);

            extractFilter->SetInput(inputImage);

            // Calculates the circles for actual axial slice
            slice_list.push_back(do_hough_on_image(extractFilter->GetOutput(), reader_original->GetOutput(), true, circles_list));

        }
        // Calculating the circles on the axial slices *circles_list* will conatin the lists of circles for axial slices *END*

        // If there is an axial slice contains zero circles, we go to the next image *STAR*
        bool zero_c_flag = false;
        auto it = circles_list.begin();
        while (it != circles_list.end())
        {
            // CirclesListType::const_iterator itCircles = (*it).begin();
            int s = (*it).size();
            // std::cout << s << std::endl;
            if (s < 1) {
                zero_c_flag = true;
                break;
            }
            it++;
        }
        if (zero_c_flag) {
            number_of_test_file++;
            continue;
        }
        // If there is an axial slice contains zero circles, we go to the next image *END*

        
        // Writing out the spine (hopefully) without corrigation *START*
        // Draws the circles on input iamge
        draw_circle(reader_original->GetOutput(), circles_list);
        std::string name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_0_corrig.nii.gz";

        writerT->SetFileName(name_of_the_file);
        writerT->SetInput(reader_original->GetOutput());
        try
        {
            writerT->Update();
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }
        // Writing out the spine (hopefully) without corrigation *END*

        // Writing out the spine (hopefully) with gloabl corrigations of the circles *START*
        corrigate_circles(circles_list);
        draw_circle(reader_original_1->GetOutput(), circles_list);

        name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_1_corrig.nii.gz";

        writerT->SetFileName(name_of_the_file);
        writerT->SetInput(reader_original_1->GetOutput());
        try
        {
            writerT->Update();
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }
        // Writing out the spine (hopefully) with gloabl corrigations of the circles *END*

        // Writing out the spine (hopefully) with local corrigations of the circles *START*
        corrigate_circles_local(circles_list);
        draw_circle(reader_original_2->GetOutput(), circles_list);

        name_of_the_file = std::to_string(number_of_test_file) + "_original_circles_2_corrig.nii.gz";

        writerT->SetFileName(name_of_the_file);
        writerT->SetInput(reader_original_2->GetOutput());
        try
        {
            writerT->Update();
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }
        // Writing out the spine (hopefully) with local corrigations of the circles *END*

        // Cleaning and labeling the ribs *START*
        ImageType::Pointer image_ribs_only = find_ribs(reader_ribs->GetOutput(), number_of_test_file, circles_list, false);

        DuplicatorType::Pointer duplicator = DuplicatorType::New();
        duplicator->SetInputImage(image_ribs_only);
        duplicator->Update();

        ImageType::Pointer image_ribs_cleaned = label_ribs_ccomponents(duplicator->GetOutput(), number_of_test_file);

        //MY LABELING
        ImageType::Pointer image_ribs_cleaned_non_labeled = find_ribs(image_ribs_cleaned, number_of_test_file, circles_list, true);

        label_ribs(image_ribs_cleaned_non_labeled, image_ribs_cleaned, number_of_test_file, circles_list);
        // Cleaning and labeling the ribs *END*


        number_of_test_file++;
    }

    return EXIT_SUCCESS;
}