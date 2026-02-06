import { useSearch } from '@/hooks/useSearch'
import DegradedBanner from '@/components/DegradedBanner'
import SearchBar from '@/components/SearchBar'
import SearchStats from '@/components/SearchStats'
import FilterPanel from '@/components/FilterPanel'
import SkillCloud from '@/components/SkillCloud'
import CompanySearch from '@/components/CompanySearch'
import JobList from '@/components/JobList'
import SimilarJobsModal from '@/components/SimilarJobsModal'

function App() {
  const {
    filters,
    setFilters,
    handleSearch,
    searchResult,
    hasSearched,
    skillCloud,
    handleSkillClick,
    health,
    similarModalOpen,
    similarJobs,
    similarLoading,
    handleFindSimilar,
    closeSimilarModal,
    companyResults,
    companyLoading,
    handleCompanySearch,
  } = useSearch()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold tracking-tight text-gray-900">
            MCF Job Search
          </h1>
        </div>
      </header>

      {/* Degraded banner */}
      <DegradedBanner show={health.data?.degraded ?? false} />

      {/* Search bar */}
      <div className="mx-auto max-w-7xl px-4 pt-6 sm:px-6 lg:px-8">
        <SearchBar onSearch={handleSearch} isLoading={searchResult.isFetching} />
      </div>

      {/* Main content: sidebar + results */}
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <aside className="w-full lg:w-80 shrink-0 space-y-6">
            <FilterPanel filters={filters} onChange={setFilters} />

            {skillCloud.data && (
              <SkillCloud
                items={skillCloud.data.items}
                onSkillClick={handleSkillClick}
              />
            )}

            <CompanySearch
              onSearch={handleCompanySearch}
              results={companyResults}
              isLoading={companyLoading}
            />
          </aside>

          {/* Main results */}
          <main className="flex-1 min-w-0">
            {searchResult.data && (
              <div className="mb-4">
                <SearchStats data={searchResult.data} />
              </div>
            )}

            <JobList
              jobs={searchResult.data?.results}
              isLoading={searchResult.isFetching}
              hasSearched={hasSearched}
              onFindSimilar={handleFindSimilar}
            />
          </main>
        </div>
      </div>

      {/* Similar jobs modal */}
      <SimilarJobsModal
        open={similarModalOpen}
        onClose={closeSimilarModal}
        jobs={similarJobs}
        isLoading={similarLoading}
        onFindSimilar={handleFindSimilar}
      />
    </div>
  )
}

export default App
